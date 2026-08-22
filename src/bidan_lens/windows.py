from __future__ import annotations

import ctypes
import sys
from contextlib import suppress

WDA_EXCLUDEFROMCAPTURE = 0x00000011


def exclude_window_from_capture(window_handle: int) -> bool:
    """Ask Windows 10 2004+ not to include the popup in screen capture."""
    if sys.platform != "win32":
        return False
    try:
        return bool(
            ctypes.windll.user32.SetWindowDisplayAffinity(
                ctypes.c_void_p(window_handle), WDA_EXCLUDEFROMCAPTURE
            )
        )
    except (AttributeError, OSError):
        return False


def enable_per_monitor_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        with suppress(AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
