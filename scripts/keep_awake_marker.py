"""Hold a Windows wakelock until a marker file is deleted.

Robust to chained processes that spawn new PIDs — `keep_awake.py` only
watches one PID, so it exits when phase1 finishes even if phase3/4
should still be running. This version watches a marker file instead.

Usage:
    touch /tmp/phase4_chain.marker
    python keep_awake_marker.py /tmp/phase4_chain.marker
    # ... long-running chain ...
    rm /tmp/phase4_chain.marker  # → keep_awake_marker.py exits
"""

from __future__ import annotations

import ctypes
import os
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python keep_awake_marker.py <marker_file>", file=sys.stderr)
        return 2
    marker = sys.argv[1]
    if not os.path.exists(marker):
        print(f"Marker {marker} does not exist", file=sys.stderr)
        return 1

    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    if not ctypes.windll.kernel32.SetThreadExecutionState(flags):
        print("SetThreadExecutionState failed", file=sys.stderr)
        return 1

    print(f"keep-awake on; watching marker {marker}", flush=True)
    while os.path.exists(marker):
        time.sleep(15)

    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print("Marker deleted; wakelock released", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
