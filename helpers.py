"""Shared helpers for resolution/frame calculations."""


def _snap_to_multiple(value: int, divisor: int) -> int:
    """Round up to nearest multiple of divisor."""
    return ((value + divisor - 1) // divisor) * divisor


def _resolution_to_dims(resolution: str) -> tuple[int, int]:
    """Parse '1920x1080' into (width, height), snapped to multiples of 64."""
    w, h = resolution.split("x")
    return _snap_to_multiple(int(w), 64), _snap_to_multiple(int(h), 64)


def _duration_to_frames(duration: float, fps: float) -> int:
    """Convert duration (seconds) to frame count, snapped to 8k+1."""
    raw = int(duration * fps)
    k = max(round((raw - 1) / 8), 1)
    return 8 * k + 1
