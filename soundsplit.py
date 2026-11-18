"""
SoundSplit - mirror whatever plays on your Windows default output device
to any number of other output devices, each with its own volume and delay.

How it works:
  * WASAPI loopback-captures the current default render device.
  * Feeds that audio into a per-target ring buffer.
  * Each target device has its own output stream that pulls from its buffer
    through an adaptive cubic resampler, which handles both the sample-rate
    difference (e.g. 44.1k -> 48k) and the slow clock drift between devices.
  * Each target keeps an adaptive safety cushion: if its buffer ever runs dry
    while music is playing, the cushion grows so it doesn't happen again.
  * Devices coming and going are handled per device; only a change of the
    Windows default device restarts the capture.

Run:  venv\\Scripts\\pythonw.exe soundsplit.py          (window + tray)
      venv\\Scripts\\pythonw.exe soundsplit.py --hidden (tray only)
"""

from __future__ import annotations

import ctypes
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
import winreg
from tkinter import ttk

import numpy as np
import pyaudiowpatch as pyaudio

APP_NAME = "SoundSplit"
HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(HERE, "settings.json")
LOG_PATH = os.path.join(HERE, "soundsplit.log")

# Audio tuning
OUT_PERIOD_S = 0.020       # output callback size. PyAudio forces Windows' *minimum*
CAP_PERIOD_S = 0.020       # latency (3 ms!), so bigger periods = bigger safety margin
START_CUSHION_MS = 80      # buffered audio kept per target (grows on dropouts)
CUSHION_STEP_MS = 40
MAX_CUSHION_MS = 400
RESYNC_MS = 150            # if off-target by more than this, jump instead of drifting
SOURCE_IDLE_S = 0.15       # no capture data for this long = nothing is playing
MAX_DELAY_MS = 1000
RING_SECONDS = 4
POLL_SECONDS = 1.0
RETRY_SECONDS = 3.0


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    if sys.stdout:
        try:
            print(line)
        except Exception:
            pass


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

class Settings:
    def __init__(self, path: str | None = SETTINGS_PATH) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.devices: dict[str, dict] = {}
        self.paused = False
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                self.devices = data.get("devices", {})
                self.paused = data.get("paused", False)
            except (OSError, ValueError):
                pass

    def get(self, name: str) -> dict:
        with self.lock:
            d = self.devices.get(name)
            if d is None:
                d = self.devices[name] = {"enabled": False, "delay_ms": 0}
            return d

    def save(self) -> None:
        if not self.path:
            return
        with self.lock:
            data = {"devices": self.devices, "paused": self.paused}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------

class Ring:
    """Thread-safe float32 frame ring buffer (single producer, single consumer)."""

    def __init__(self, frames: int, channels: int) -> None:
        self.buf = np.zeros((frames, channels), dtype=np.float32)
        self.cap = frames
        self.r = 0
        self.n = 0
        self.lock = threading.Lock()

    def available(self) -> int:
        return self.n

    def write(self, data: np.ndarray) -> None:
        with self.lock:
            m = len(data)
            if m >= self.cap:
                data = data[-self.cap:]
                m = self.cap
            overflow = self.n + m - self.cap
            if overflow > 0:          # drop oldest
                self.r = (self.r + overflow) % self.cap
                self.n -= overflow
            w = (self.r + self.n) % self.cap
            first = min(m, self.cap - w)
            self.buf[w:w + first] = data[:first]
            if first < m:
                self.buf[:m - first] = data[first:]
            self.n += m

    def peek(self, m: int) -> np.ndarray:
        with self.lock:
            m = min(m, self.n)
            first = min(m, self.cap - self.r)
            if first == m:
                return self.buf[self.r:self.r + m].copy()
            return np.concatenate([self.buf[self.r:], self.buf[:m - first]])

    def skip(self, m: int) -> None:
        with self.lock:
            m = min(m, self.n)
            self.r = (self.r + m) % self.cap
            self.n -= m


def match_channels(x: np.ndarray, ch: int) -> np.ndarray:
    have = x.shape[1]
    if have == ch:
        return x
    if ch == 1:
        return x[:, :2].mean(axis=1, keepdims=True)
    if have == 1:
        return np.repeat(x, ch, axis=1)
    if have > ch:
        return x[:, :ch]
    out = np.zeros((len(x), ch), dtype=np.float32)
    out[:, :have] = x
    return out


class Mirror:
    """One target output device fed from the loopback capture."""

    def __init__(self, pa: pyaudio.PyAudio, info: dict, src_rate: int,
                 settings: Settings, level: float = 1.0) -> None:
        self.name = info["name"]
        self.settings = settings
        self.level = level                    # 1.0 normally; tests use 0.0
        self.src_rate = src_rate
        self.dst_rate = int(info["defaultSampleRate"])
        self.ch = min(2, int(info["maxOutputChannels"])) or 2
        self.ring = Ring(src_rate * RING_SECONDS, self.ch)
        self.pos = 0.0                        # fractional read position
        self.hist = np.zeros((1, self.ch), dtype=np.float32)
        self.fill_ema = 0.0
        self.priming = True
        self.gain = 0.0
        self.last_push = 0.0
        self.cushion_ms = START_CUSHION_MS
        self.dropouts = 0                     # our buffer ran dry mid-music
        self.xruns = 0                        # Windows reported the device starved
        self.status = "starting"
        self.error: str | None = None
        self.stream = pa.open(
            format=pyaudio.paFloat32,
            channels=self.ch,
            rate=self.dst_rate,
            output=True,
            output_device_index=info["index"],
            frames_per_buffer=int(self.dst_rate * OUT_PERIOD_S),
            stream_callback=self._callback,
        )

    # producer side (capture thread)
    def push(self, frames: np.ndarray) -> None:
        self.ring.write(match_channels(frames, self.ch))
        self.last_push = time.monotonic()

    # consumer side (output thread)
    def _callback(self, _in, n, _time, status):
        if status & pyaudio.paOutputUnderflow and self.status == "playing":
            self.xruns += 1
        try:
            return self._render(n).tobytes(), pyaudio.paContinue
        except Exception as e:  # never let the audio thread die silently
            self.error = repr(e)
            log(f"[{self.name}] callback error: {traceback.format_exc()}")
            return np.zeros((n, self.ch), np.float32).tobytes(), pyaudio.paContinue

    def _underrun(self) -> None:
        # Only a real dropout if music was still arriving; if the source went
        # quiet (paused / nothing playing) running dry is expected.
        if self.status == "playing" and time.monotonic() - self.last_push < SOURCE_IDLE_S:
            self.dropouts += 1
            if self.cushion_ms < MAX_CUSHION_MS:
                self.cushion_ms = min(MAX_CUSHION_MS, self.cushion_ms + CUSHION_STEP_MS)
                log(f"[{self.name}] dropout -> cushion {self.cushion_ms} ms")
        self.priming = True
        self.gain = 0.0                        # fade back in when resuming
        self.status = "waiting for audio"

    def _render(self, n: int) -> np.ndarray:
        cfg = self.settings.get(self.name)
        silence = np.zeros((n, self.ch), dtype=np.float32)
        sr = self.src_rate
        target = (self.cushion_ms + cfg["delay_ms"]) / 1000 * sr
        resync = RESYNC_MS / 1000 * sr
        fill = self.ring.available()

        if self.priming:
            if fill >= target:
                self.priming = False
                self.fill_ema = fill
                self.pos = 0.0
            else:
                self.status = "waiting for audio"
                return silence

        # big jumps (delay slider moved, long stall): resync immediately
        if fill > target + resync:
            self.ring.skip(int(fill - target))
            fill = self.ring.available()
            self.fill_ema = fill
        elif fill < target - resync and fill < n * 2:
            self._underrun()
            return silence

        self.fill_ema = 0.98 * self.fill_ema + 0.02 * fill
        err_s = (self.fill_ema - target) / sr
        corr = max(-0.003, min(0.003, err_s * 0.04))     # <= 0.3% speed nudge
        ratio = (sr / self.dst_rate) * (1.0 + corr)

        last = self.pos + ratio * (n - 1)
        need = int(last) + 3
        if self.ring.available() < need:
            self._underrun()
            return silence

        w = np.concatenate([self.hist, self.ring.peek(need)])   # w[k+1] == frame k
        p = self.pos + ratio * np.arange(n)
        i = p.astype(np.int64)
        t = (p - i)[:, None].astype(np.float32)
        y0, y1, y2, y3 = w[i], w[i + 1], w[i + 2], w[i + 3]
        # Catmull-Rom cubic interpolation between y1 and y2
        out = y1 + 0.5 * t * (y2 - y0 + t * (2 * y0 - 5 * y1 + 4 * y2 - y3
                                             + t * (3 * (y1 - y2) + y3 - y0)))

        consumed = int(self.pos + ratio * n)
        self.hist = w[consumed:consumed + 1].copy()
        self.ring.skip(consumed)
        self.pos = self.pos + ratio * n - consumed

        new_gain = self.level   # loudness itself is the device's own Windows volume
        if new_gain != self.gain:
            out = out * np.linspace(self.gain, new_gain, n, dtype=np.float32)[:, None]
            self.gain = new_gain
        elif new_gain != 1.0:
            out = out * new_gain
        self.status = "playing"
        return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)

    def describe(self) -> str:
        if self.status != "playing":
            return self.status
        s = f"playing · {self.cushion_ms} ms"
        drops = self.dropouts + self.xruns
        if drops:
            s += f" · {drops} dropout{'s' if drops != 1 else ''}"
        return s

    def close(self) -> None:
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Windows Core Audio helpers (device list, names, per-device volume)
# --------------------------------------------------------------------------

def device_enumerator():
    from pycaw.pycaw import AudioUtilities
    return AudioUtilities.GetDeviceEnumerator()


def active_outputs(enum):
    from pycaw.constants import DEVICE_STATE, EDataFlow
    c = enum.EnumAudioEndpoints(EDataFlow.eRender.value, DEVICE_STATE.ACTIVE.value)
    return [c.Item(i) for i in range(c.GetCount())]


def friendly_name(dev) -> str | None:
    """Read ONLY PKEY_Device_FriendlyName. (pycaw's CreateDevice decodes every
    property of the device, which can crash on devices mid-(dis)connect.)"""
    from comtypes import GUID
    from pycaw.api.mmdeviceapi.depend.structures import PROPERTYKEY
    key = PROPERTYKEY(GUID("{a45c254e-df1c-4efd-8020-67d146a850e0}"), 14)
    try:
        pv = dev.OpenPropertyStore(0).GetValue(key)   # STGM_READ
    except Exception:
        return None
    try:
        return pv.union.pwszVal if pv.vt == 31 else None  # VT_LPWSTR
    finally:
        pv.clear()


class Engine:
    """Owns PortAudio, the loopback capture and the per-device mirrors.
    All PortAudio/COM work happens on the engine thread (run())."""

    def __init__(self, settings: Settings, level: float = 1.0) -> None:
        self.settings = settings
        self.level = level
        self.lock = threading.RLock()
        self.pa: pyaudio.PyAudio | None = None
        self.capture = None
        self.mirrors: dict[str, Mirror] = {}
        self.source_id: str | None = None
        self.source_name: str | None = None
        self.targets: list[str] = []          # other active outputs, by name
        self.errors: dict[str, str] = {}
        self.version = 0                      # bumps whenever targets change
        self._outputs: dict[str, dict] = {}   # name -> PortAudio device info
        self._src_rate: int | None = None
        self._src_ch = 2
        self._push_targets: tuple[Mirror, ...] = ()
        self._names: dict[str, str] = {}      # endpoint id -> friendly name
        self._retry_at: dict[str, float] = {}
        self._need_rescan = False
        self._last_rebuild = 0.0
        self._stop = threading.Event()
        self._wake = threading.Event()

    def poke(self) -> None:
        """Ask the engine to re-check enabled devices now (called from the UI)."""
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def run(self) -> None:
        import comtypes
        comtypes.CoInitialize()
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception:
                log("poll error: " + traceback.format_exc())
            self._wake.wait(POLL_SECONDS)
            self._wake.clear()
        self.teardown()

    # ---- polling ----
    def poll(self) -> None:
        from pycaw.constants import ERole
        enum = device_enumerator()
        try:
            default_id = enum.GetDefaultAudioEndpoint(0, ERole.eMultimedia.value).GetId()
        except Exception:
            default_id = None
        present: dict[str, str] = {}
        for dev in active_outputs(enum):
            dev_id = dev.GetId()
            name = self._names.get(dev_id) or friendly_name(dev)
            if name:
                self._names[dev_id] = name
                present[dev_id] = name

        targets = sorted(n for i, n in present.items() if i != default_id)
        if targets != self.targets:
            with self.lock:
                self.targets = targets
                self.version += 1

        now = time.monotonic()
        capture_dead = self.capture is not None and not self.capture.is_active()
        if default_id != self.source_id:
            if self.source_id is not None:
                log(f"default device changed -> {present.get(default_id)}")
            self.rebuild(default_id, present.get(default_id))
        elif default_id and (self.capture is None or capture_dead or self._need_rescan) \
                and now - self._last_rebuild > RETRY_SECONDS:
            log("capture stopped" if capture_dead else "rescanning devices")
            self.rebuild(default_id, present.get(default_id))
        self.sync_mirrors()

    def teardown(self) -> None:
        with self.lock:
            self._push_targets = ()
            for m in self.mirrors.values():
                m.close()
            self.mirrors = {}
            if self.capture is not None:
                try:
                    self.capture.stop_stream()
                    self.capture.close()
                except Exception:
                    pass
                self.capture = None
            if self.pa is not None:
                try:
                    self.pa.terminate()
                except Exception:
                    pass
                self.pa = None
            self._src_rate = None

    def rebuild(self, default_id: str | None, default_name: str | None) -> None:
        """Fresh PortAudio (= fresh device list) + capture of the default device."""
        self.teardown()
        time.sleep(0.3)  # let Windows settle after a device switch
        with self.lock:
            self._last_rebuild = time.monotonic()
            self._need_rescan = False
            self._retry_at = {}
            self.errors = {}
            self.source_id = default_id
            self.source_name = default_name
            if default_id is None:
                return
            pa = self.pa = pyaudio.PyAudio()
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            pa_default = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])["name"]
            src_name = default_name or pa_default

            loopback = None
            self._outputs = {}
            for i in range(pa.get_device_count()):
                d = pa.get_device_info_by_index(i)
                if d["hostApi"] != wasapi["index"]:
                    continue
                if d.get("isLoopbackDevice"):
                    if d["name"] == f"{src_name} [Loopback]":
                        loopback = d
                elif d["maxOutputChannels"] > 0 and d["name"] != src_name:
                    self._outputs[d["name"]] = d
            log(f"source={src_name!r} outputs={list(self._outputs)}")
            if loopback is None:
                self.errors["__source__"] = "can't capture the default device"
                log(f"no loopback device for {src_name!r}")
                return

            self._src_rate = int(loopback["defaultSampleRate"])
            self._src_ch = src_ch = int(loopback["maxInputChannels"])

            def on_capture(data, frames, _time, _status):
                targets = self._push_targets
                if targets:
                    x = np.frombuffer(data, dtype=np.float32).reshape(frames, src_ch)
                    for m in targets:
                        m.push(x)
                return None, pyaudio.paContinue

            try:
                self.capture = pa.open(
                    format=pyaudio.paFloat32,
                    channels=src_ch,
                    rate=self._src_rate,
                    input=True,
                    input_device_index=loopback["index"],
                    frames_per_buffer=int(self._src_rate * CAP_PERIOD_S),
                    stream_callback=on_capture,
                )
            except Exception as e:
                self.errors["__source__"] = f"capture failed: {e}"
                log(f"capture open failed: {e}")

    def sync_mirrors(self) -> None:
        """Open streams for ticked devices, close the rest. Devices are handled
        individually so one device coming/going doesn't interrupt the others."""
        with self.lock:
            if self.pa is None or self._src_rate is None:
                return
            now = time.monotonic()
            want = set() if self.settings.paused else {
                n for n in self.targets if self.settings.get(n)["enabled"]}

            for name, m in list(self.mirrors.items()):
                if name not in want:
                    m.close()
                    del self.mirrors[name]
                elif not m.stream.is_active():
                    log(f"[{name}] stream stopped -> reopening")
                    m.close()
                    del self.mirrors[name]

            for name in want - set(self.mirrors):
                if now < self._retry_at.get(name, 0):
                    continue
                info = self._outputs.get(name)
                if info is None:              # connected after our last device scan
                    self._need_rescan = True
                    continue
                try:
                    self.mirrors[name] = Mirror(self.pa, info, self._src_rate,
                                                self.settings, self.level)
                    self.errors.pop(name, None)
                    log(f"[{name}] opened ({info['defaultSampleRate']:.0f} Hz)")
                except Exception as e:
                    self.errors[name] = str(e)
                    self._retry_at[name] = now + RETRY_SECONDS
                    self._need_rescan = True  # device info may be stale (reconnected)
                    log(f"[{name}] open failed: {e}")

            for name in list(self.errors):
                if name != "__source__" and name not in want:
                    del self.errors[name]
            self._push_targets = tuple(self.mirrors.values())

    def snapshot(self):
        with self.lock:
            status = {}
            for name in self.targets:
                m = self.mirrors.get(name)
                if m is not None:
                    status[name] = "error (see log)" if m.error else m.describe()
                elif name in self.errors:
                    status[name] = "error: " + self.errors[name][:40]
                elif self.settings.paused or not self.settings.get(name)["enabled"]:
                    status[name] = "off"
                else:
                    status[name] = "connecting…"
            return (self.version, self.source_name, list(self.targets), status,
                    self.errors.get("__source__"))


# --------------------------------------------------------------------------
# Per-device Windows volume (the same value the Windows volume slider shows)
# --------------------------------------------------------------------------

class DeviceVolumes:
    """Reads/sets each output device's own Windows endpoint volume by name.
    Must be used from a single (COM-initialised) thread - the UI thread."""

    def __init__(self) -> None:
        import comtypes
        comtypes.CoInitialize()
        self._ifaces: dict = {}
        self._last_refresh = 0.0

    def refresh(self) -> None:
        from ctypes import POINTER, cast
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import IAudioEndpointVolume
        self._last_refresh = time.monotonic()
        self._ifaces = {}
        try:
            devices = active_outputs(device_enumerator())
        except Exception as e:
            log(f"volume refresh failed: {e}")
            return
        for d in devices:
            try:
                name = friendly_name(d)
                if name:
                    v = d.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                    self._ifaces[name] = cast(v, POINTER(IAudioEndpointVolume))
            except Exception:
                pass

    def _call(self, name: str, fn):
        """Run fn(iface); if Windows invalidated the endpoint (device reset /
        reconnected), grab fresh handles once and retry."""
        for attempt in (0, 1):
            v = self._ifaces.get(name)
            if v is not None:
                try:
                    return fn(v)
                except Exception:
                    pass
            if attempt == 0 and time.monotonic() - self._last_refresh > 1.0:
                self.refresh()
            else:
                break
        return None

    def get(self, name: str) -> int | None:
        return self._call(name, lambda v: round(v.GetMasterVolumeLevelScalar() * 100))

    def set(self, name: str, pct: int) -> bool:
        def do(v):
            v.SetMasterVolumeLevelScalar(max(0, min(100, pct)) / 100.0, None)
            if pct > 0 and v.GetMute():
                v.SetMute(0, None)
            return True
        return self._call(name, do) is not None


# --------------------------------------------------------------------------
# Start with Windows
# --------------------------------------------------------------------------

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def startup_command() -> str:
    exe = sys.executable
    if exe.lower().endswith("python.exe"):
        exe = exe[:-len("python.exe")] + "pythonw.exe"
    return f'"{exe}" "{os.path.abspath(__file__)}" --hidden'


def startup_enabled() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            winreg.QueryValueEx(k, APP_NAME)
            return True
    except OSError:
        return False


def set_startup(on: bool) -> None:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
        if on:
            winreg.SetValueEx(k, APP_NAME, 0, winreg.REG_SZ, startup_command())
        else:
            try:
                winreg.DeleteValue(k, APP_NAME)
            except OSError:
                pass


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def status_color(s: str) -> str:
    if s.startswith("playing"):
        return "#b58900" if "dropout" in s else "#2e9e4f"
    if s.startswith(("waiting", "connecting", "starting")):
        return "#8a7a3a"
    if s == "off":
        return "#888888"
    return "#c0392b"


class App:
    def __init__(self, start_hidden: bool) -> None:
        self.settings = Settings()
        self.engine = Engine(self.settings)
        self.cmds: queue.Queue = queue.Queue()
        self.version_shown = -1
        self.rows: dict[str, dict] = {}
        self._save_job = None
        self._ticks = 0

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        self.volumes = DeviceVolumes()
        self.root = tk.Tk()
        self.root.title(APP_NAME)
        self.root.minsize(640, 200)
        self.root.protocol("WM_DELETE_WINDOW", self.hide)

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        self.source_var = tk.StringVar(value="Source: …")
        self.source_lbl = ttk.Label(outer, textvariable=self.source_var,
                                    font=("Segoe UI", 10, "bold"))
        self.source_lbl.pack(anchor="w")
        ttk.Label(outer, foreground="#777",
                  text="Everything playing on your Windows default device is copied to the "
                       "devices ticked below.").pack(anchor="w", pady=(2, 10))

        self.list_frame = ttk.Frame(outer)
        self.list_frame.pack(fill="both", expand=True)

        bottom = ttk.Frame(outer)
        bottom.pack(fill="x", pady=(12, 0))
        self.paused_var = tk.BooleanVar(value=self.settings.paused)
        ttk.Checkbutton(bottom, text="Pause all", variable=self.paused_var,
                        command=self.toggle_pause).pack(side="left")
        self.startup_var = tk.BooleanVar(value=startup_enabled())
        ttk.Checkbutton(bottom, text="Start with Windows", variable=self.startup_var,
                        command=lambda: set_startup(self.startup_var.get())
                        ).pack(side="left", padx=16)
        ttk.Label(bottom, foreground="#777",
                  text="Delay holds a device back to line it up with a slower one."
                  ).pack(side="right")

        self.icon = self.make_tray()
        if start_hidden:
            self.root.withdraw()

        threading.Thread(target=self.engine.run, daemon=True, name="engine").start()
        self.root.after(200, self.tick)

    # ---- tray ----
    def make_tray(self):
        try:
            import pystray
            from PIL import Image, ImageDraw
        except ImportError:
            return None
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        dr = ImageDraw.Draw(img)
        dr.ellipse((4, 14, 40, 50), fill=(64, 156, 255))
        dr.ellipse((24, 14, 60, 50), fill=(46, 158, 79))
        dr.ellipse((24, 14, 40, 50), fill=(30, 110, 140))
        menu = pystray.Menu(
            pystray.MenuItem("Show", lambda: self.cmds.put("show"), default=True),
            pystray.MenuItem("Pause all", lambda: self.cmds.put("pause"),
                             checked=lambda _: self.settings.paused),
            pystray.MenuItem("Quit", lambda: self.cmds.put("quit")),
        )
        icon = pystray.Icon(APP_NAME, img, APP_NAME, menu)
        icon.run_detached()
        return icon

    def show(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def hide(self) -> None:
        if self.icon is None:
            self.quit()
        else:
            self.root.withdraw()

    def quit(self) -> None:
        self.engine.stop()
        if self.icon is not None:
            self.icon.stop()
        self.flush_save()
        self.root.destroy()

    def toggle_pause(self) -> None:
        self.settings.paused = self.paused_var.get()
        self.engine.poke()
        self.schedule_save()

    # ---- settings save (debounced) ----
    def schedule_save(self) -> None:
        if self._save_job:
            self.root.after_cancel(self._save_job)
        self._save_job = self.root.after(500, self.flush_save)

    def flush_save(self) -> None:
        self._save_job = None
        try:
            self.settings.save()
        except OSError as e:
            log(f"could not save settings: {e}")

    # ---- device rows ----
    def build_rows(self, targets: list[str]) -> None:
        for w in self.list_frame.winfo_children():
            w.destroy()
        self.rows = {}
        self.volumes.refresh()
        if not targets:
            ttk.Label(self.list_frame, foreground="#777",
                      text="No other output devices connected.").grid(row=0, column=0, sticky="w")
            return
        for c, h in enumerate(["Device", "Volume", "", "Delay", "", "Status"]):
            ttk.Label(self.list_frame, text=h, foreground="#777").grid(
                row=0, column=c, sticky="w", padx=4)
        for r, name in enumerate(targets, start=1):
            cfg = self.settings.get(name)
            en = tk.BooleanVar(value=cfg["enabled"])
            cur = self.volumes.get(name)
            vol = tk.IntVar(value=cur if cur is not None else 0)
            dly = tk.IntVar(value=cfg["delay_ms"])
            vol_lbl = ttk.Label(self.list_frame, width=4,
                                text=f"{cur}" if cur is not None else "?")
            dly_lbl = ttk.Label(self.list_frame, width=7, text=f"{cfg['delay_ms']} ms")
            status = ttk.Label(self.list_frame, width=28)

            def on_en(name=name, en=en):
                self.settings.get(name)["enabled"] = en.get()
                self.engine.poke()
                self.schedule_save()

            def on_vol(_v, name=name, lbl=vol_lbl):
                v = int(float(_v))
                lbl.config(text=f"{v}" if self.volumes.set(name, v) else "?")

            def on_dly(_v, name=name, lbl=dly_lbl):
                v = int(round(float(_v) / 10) * 10)
                self.settings.get(name)["delay_ms"] = v
                lbl.config(text=f"{v} ms")
                self.schedule_save()

            ttk.Checkbutton(self.list_frame, text=name, variable=en, command=on_en).grid(
                row=r, column=0, sticky="w", padx=4, pady=4)
            ttk.Scale(self.list_frame, from_=0, to=100, variable=vol, length=130,
                      command=on_vol).grid(row=r, column=1, padx=4)
            vol_lbl.grid(row=r, column=2, sticky="w")
            ttk.Scale(self.list_frame, from_=0, to=MAX_DELAY_MS, variable=dly, length=130,
                      command=on_dly).grid(row=r, column=3, padx=4)
            dly_lbl.grid(row=r, column=4, sticky="w")
            status.grid(row=r, column=5, sticky="w", padx=4)
            self.rows[name] = {"status": status, "vol": vol, "vol_lbl": vol_lbl}
        self.list_frame.columnconfigure(0, weight=1)

    def tick(self) -> None:
        try:
            while True:
                cmd = self.cmds.get_nowait()
                if cmd == "show":
                    self.show()
                elif cmd == "pause":
                    self.paused_var.set(not self.settings.paused)
                    self.toggle_pause()
                elif cmd == "quit":
                    self.quit()
                    return
        except queue.Empty:
            pass

        try:
            self.refresh_ui()
        except Exception:
            log("ui error: " + traceback.format_exc())
        self.root.after(250, self.tick)

    def refresh_ui(self) -> None:
        version, source, targets, status, source_err = self.engine.snapshot()
        if source_err:
            self.source_var.set(f"Source: {source or '?'}  —  {source_err}")
        elif source:
            self.source_var.set(f"Source (Windows default): {source}")
        else:
            self.source_var.set("Source: no default output device")
        if version != self.version_shown:
            self.version_shown = version
            self.build_rows(targets)
        self._ticks += 1
        sync_vol = self._ticks % 4 == 0   # ~1 s: follow volume changes made elsewhere
        for name, row in self.rows.items():
            s = status.get(name, "")
            row["status"].config(text=s, foreground=status_color(s))
            if sync_vol:
                v = self.volumes.get(name)
                if v is not None and v != row["vol"].get():
                    row["vol"].set(v)
                    row["vol_lbl"].config(text=f"{v}")

    def run(self) -> None:
        self.root.mainloop()


def single_instance() -> bool:
    ctypes.windll.kernel32.CreateMutexW(None, False, "Global\\SoundSplitSingleInstance")
    return ctypes.windll.kernel32.GetLastError() != 183  # ERROR_ALREADY_EXISTS


def tune_process() -> None:
    # Audio callbacks are Python code: keep the interpreter responsive to them.
    sys.setswitchinterval(0.001)
    try:  # ABOVE_NORMAL_PRIORITY_CLASS
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),
                                                0x00008000)
    except Exception:
        pass


if __name__ == "__main__":
    import faulthandler
    _crash_log = open(LOG_PATH, "a", encoding="utf-8")
    faulthandler.enable(file=_crash_log, all_threads=True)  # native crash -> stack in log
    if not single_instance():
        ctypes.windll.user32.MessageBoxW(None, "SoundSplit is already running (check the tray).",
                                         APP_NAME, 0x40)
        sys.exit(0)
    tune_process()
    log("---- SoundSplit started ----")
    App(start_hidden="--hidden" in sys.argv).run()
