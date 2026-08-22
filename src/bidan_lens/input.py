from __future__ import annotations

import threading
from collections.abc import Callable


class InputMonitor:
    def __init__(
        self,
        activation_key: str,
        previous_result_key: str,
        next_result_key: str,
        on_previous: Callable[[], None],
        on_next: Callable[[], None],
    ) -> None:
        self.activation_key = self._canonical_name(activation_key)
        self.previous_result_keys = self._parse_chord(previous_result_key)
        self.next_result_keys = self._parse_chord(next_result_key)
        self.on_previous = on_previous
        self.on_next = on_next
        self._held: set[str] = set()
        self._lock = threading.Lock()
        self._keyboard = None
        self._mouse = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self.activation_key in self._held

    def start(self) -> None:
        from pynput import keyboard, mouse

        self._keyboard = keyboard.Listener(on_press=self._press, on_release=self._release)
        self._mouse = mouse.Listener(on_scroll=self._scroll)
        self._keyboard.start()
        self._mouse.start()

    def stop(self) -> None:
        if self._keyboard:
            self._keyboard.stop()
        if self._mouse:
            self._mouse.stop()

    @staticmethod
    def _name(key: object) -> str:
        character = getattr(key, "char", None)
        if character:
            return str(character).lower()
        name = getattr(key, "name", None)
        return InputMonitor._canonical_name(str(name or key).replace("Key.", ""))

    @staticmethod
    def _canonical_name(name: str) -> str:
        lowered = name.strip().lower()
        aliases = {
            "control": "ctrl",
            "ctrl_l": "ctrl",
            "ctrl_r": "ctrl",
            "shift_l": "shift",
            "shift_r": "shift",
            "alt_l": "alt",
            "alt_r": "alt",
        }
        return aliases.get(lowered, lowered)

    @classmethod
    def _parse_chord(cls, chord: str) -> frozenset[str]:
        return frozenset(cls._canonical_name(part) for part in chord.split("+") if part.strip())

    def _press(self, key: object) -> None:
        name = self._name(key)
        with self._lock:
            already_held = name in self._held
            self._held.add(name)
        if already_held:
            return
        if self._held == self.previous_result_keys:
            self.on_previous()
        elif self._held == self.next_result_keys:
            self.on_next()

    def _release(self, key: object) -> None:
        with self._lock:
            self._held.discard(self._name(key))

    def _scroll(self, _x: int, _y: int, _dx: int, dy: int) -> None:
        if dy > 0:
            self.on_previous()
        elif dy < 0:
            self.on_next()
