from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    image: Image.Image
    origin: tuple[int, int]


class ScreenCapture:
    def __init__(self) -> None:
        import mss

        self._capture = mss.mss()

    def around(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        bounds: tuple[int, int, int, int] | None = None,
    ) -> CapturedFrame:
        if bounds is None:
            virtual = self._capture.monitors[0]
            bounds = (
                virtual["left"],
                virtual["top"],
                virtual["left"] + virtual["width"],
                virtual["top"] + virtual["height"],
            )
        left = max(bounds[0], min(x - width // 2, bounds[2] - width))
        top = max(bounds[1], min(y - height // 2, bounds[3] - height))
        monitor = {
            "left": left,
            "top": top,
            "width": min(width, bounds[2] - left),
            "height": min(height, bounds[3] - top),
        }
        shot = self._capture.grab(monitor)
        # Copy into a PIL-owned buffer so the MSS frame can be released immediately.
        image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX").copy()
        return CapturedFrame(image, (left, top))

    def close(self) -> None:
        self._capture.close()
