"""Damasio Hardware Interoception & Homeostatic Neuromodulators (Spec #90, <= 140 LOC)."""
from __future__ import annotations

from collections import deque
import ctypes, shutil, time
from dataclasses import dataclass
from typing import Any

_U8, _U32, _U64 = ctypes.c_ubyte, ctypes.c_uint32, ctypes.c_uint64


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", _U8), ("BatteryFlag", _U8), ("BatteryLifePercent", _U8),
                ("SystemStatusFlag", _U8), ("BatteryLifeTime", _U32), ("BatteryFullLifeTime", _U32)]

class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [("dwLength", _U32), ("dwMemoryLoad", _U32), ("ullTotalPhys", _U64),
                ("ullAvailPhys", _U64), ("ullTotalPageFile", _U64), ("ullAvailPageFile", _U64),
                ("ullTotalVirtual", _U64), ("ullAvailVirtual", _U64), ("ullAvailExtendedVirtual", _U64)]

class _FileTime(ctypes.Structure):
    _fields_ = [("dwLowDateTime", _U32), ("dwHighDateTime", _U32)]


@dataclass
class HomeostaticState:
    ac_plugged: bool; battery_pct: float; cpu_load_pct: float; mem_free_gb: float
    disk_e_free_gb: float; disk_e_usage_pct: float; cert_pass_rate: float; contradiction_rate: float
    da: float; ne: float; ach: float; serotonin: float; timestamp: float

    @property
    def ht5(self) -> float:
        return self.serotonin


class HomeostaticInteroceptor:
    """Samples Windows Win32 hardware telemetry + epistemic feedback into DA/NE/ACh/5-HT."""

    def __init__(self, drive_path: str = ".", window: int = 32, telemetry_override: dict[str, Any] | None = None) -> None:
        self.drive_path = drive_path
        self._certs: deque[int] = deque(maxlen=max(4, window))
        self._contras: deque[int] = deque(maxlen=max(4, window))
        self._da, self._ne, self._ach = 0.50, 0.20, 0.50
        self._override: dict[str, Any] | None = dict(telemetry_override) if telemetry_override else None
        self._k32 = getattr(ctypes, "windll", None).kernel32 if hasattr(ctypes, "windll") else None
        self._prev_times: tuple[int, int, int] = self._read_sys_times()
        self._last_cpu_pct, self.last_state = 0.0, None

    def set_telemetry_override(self, override: dict[str, Any] | None) -> None:
        self._override = dict(override) if override is not None else None

    def _read_sys_times(self) -> tuple[int, int, int]:
        if self._k32 is None:
            return (0, 0, 0)
        fi, fk, fu = _FileTime(), _FileTime(), _FileTime()
        if not self._k32.GetSystemTimes(ctypes.byref(fi), ctypes.byref(fk), ctypes.byref(fu)):
            return (0, 0, 0)
        return ((fi.dwHighDateTime << 32) | fi.dwLowDateTime,
                (fk.dwHighDateTime << 32) | fk.dwLowDateTime,
                (fu.dwHighDateTime << 32) | fu.dwLowDateTime)

    def _probe_hardware(self) -> tuple[bool, float, float, float, float, float]:
        ac, bat, mem_gb, cpu_pct = True, 100.0, 4.0, self._last_cpu_pct
        if self._k32 is not None:
            sps = _SystemPowerStatus()
            if self._k32.GetSystemPowerStatus(ctypes.byref(sps)):
                ac, bat = (sps.ACLineStatus != 0), (100.0 if sps.BatteryLifePercent == 255 else float(sps.BatteryLifePercent))
            mse = _MemoryStatusEx(dwLength=ctypes.sizeof(_MemoryStatusEx))
            if self._k32.GlobalMemoryStatusEx(ctypes.byref(mse)):
                mem_gb = mse.ullAvailPhys / 1073741824.0
            cur = self._read_sys_times()
            tot = (cur[1] - self._prev_times[1]) + (cur[2] - self._prev_times[2])
            if tot > 0:
                cpu_pct = max(0.0, min(100.0, 100.0 * max(0, tot - (cur[0] - self._prev_times[0])) / tot))
                self._last_cpu_pct, self._prev_times = cpu_pct, cur
        try:
            u = shutil.disk_usage(self.drive_path)
            e_free_gb, e_use_pct = u.free / 1073741824.0, 100.0 * u.used / max(1, u.total)
        except OSError:
            e_free_gb, e_use_pct = 0.0, 100.0
        if self._override:
            ov = self._override
            ac, bat = bool(ov.get("ac_plugged", ac)), float(ov.get("battery_pct", bat))
            cpu_pct, mem_gb = float(ov.get("cpu_load_pct", cpu_pct)), float(ov.get("mem_free_gb", mem_gb))
            e_free_gb, e_use_pct = float(ov.get("disk_e_free_gb", e_free_gb)), float(ov.get("disk_e_usage_pct", e_use_pct))
        return ac, bat, cpu_pct, mem_gb, e_free_gb, e_use_pct

    def sample(self) -> HomeostaticState:
        ac, bat, cpu_pct, mem_gb, e_free_gb, e_use_pct = self._probe_hardware()
        c_rate = (sum(self._certs) / len(self._certs)) if self._certs else 0.50
        x_rate = (sum(self._contras) / len(self._contras)) if self._contras else 0.0
        ht5 = (0.70 - (0.30 if not ac else 0.0) - (0.25 if mem_gb < 1.5 else (-0.10 if mem_gb > 3.0 else 0.0))
               + (0.15 if (ac and cpu_pct < 15.0) else (-0.15 if cpu_pct > 75.0 else 0.0))
               - (0.15 if e_use_pct >= 85.0 else 0.0) - 0.15 * x_rate)
        st = HomeostaticState(
            ac, round(bat, 1), round(cpu_pct, 2), round(mem_gb, 3), round(e_free_gb, 2), round(e_use_pct, 2),
            round(c_rate, 4), round(x_rate, 4), round(_clamp01(self._da), 4), round(_clamp01(self._ne), 4),
            round(_clamp01(self._ach), 4), round(_clamp01(ht5), 4), time.time(),
        )
        self.last_state = st
        return st

    def update_feedback(self, cert_ok: bool, contradiction: bool, surprise: float) -> HomeostaticState:
        s = _clamp01(surprise)
        self._certs.append(1 if cert_ok else 0)
        self._contras.append(1 if contradiction else 0)
        self._da = _clamp01(0.75 * self._da + 0.25 * (0.95 if cert_ok else (0.15 if contradiction else 0.45)))
        self._ne = _clamp01(0.65 * self._ne + 0.35 * max(s, 0.90 if contradiction else 0.0))
        self._ach = _clamp01(0.70 * self._ach + 0.30 * max(s, 0.85 if contradiction else 0.30))
        return self.sample()

    def sleep_allowed(self, idle_sec: float = 0.0, min_idle_sec: float = 600.0,
                      max_cpu_pct: float = 15.0, min_ram_gb: float = 1.5, max_disk_pct: float = 85.0) -> bool:
        st = self.sample()
        return (st.ac_plugged and st.cpu_load_pct < max_cpu_pct and st.mem_free_gb > min_ram_gb
                and st.disk_e_usage_pct < max_disk_pct and idle_sec >= min_idle_sec)

    def recommended_branch_k(self, default_k: int = 4) -> int:
        st = self.sample()
        k = max(1, int(default_k))
        if not st.ac_plugged:
            k = max(1, k // 2)
        if st.mem_free_gb < 1.5:
            k = max(1, min(k, 1 if not st.ac_plugged else max(1, k // 2)))
        return max(1, k - 1) if st.cpu_load_pct > 85.0 else k
