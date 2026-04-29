"""Prevent Windows sleep while a long-running task is active.

Calls Windows' SetThreadExecutionState with ES_CONTINUOUS |
ES_SYSTEM_REQUIRED so the OS won't enter sleep. Exits when the watched
PID no longer exists, which automatically clears the wakelock.

Usage:
    python keep_awake.py <pid>
"""

from __future__ import annotations

import ctypes
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def pid_alive(pid: int) -> bool:
    """Return True if a process with this PID is alive on Windows."""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, pid
    )
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(
            handle, ctypes.byref(exit_code)
        )
        STILL_ACTIVE = 259
        return exit_code.value == STILL_ACTIVE
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python keep_awake.py <pid>", file=sys.stderr)
        return 2
    pid = int(sys.argv[1])
    if not pid_alive(pid):
        print(f"PID {pid} not alive at start", file=sys.stderr)
        return 1

    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    if not ctypes.windll.kernel32.SetThreadExecutionState(flags):
        print("SetThreadExecutionState failed", file=sys.stderr)
        return 1

    print(f"keep-awake on for PID {pid}", flush=True)
    while pid_alive(pid):
        time.sleep(30)

    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print(f"PID {pid} exited; wakelock released", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
