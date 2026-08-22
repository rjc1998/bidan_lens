from bidan_lens.input import InputMonitor


class Key:
    def __init__(self, name):
        self.name = name


def test_configurable_navigation_chords_are_canonicalized() -> None:
    calls = []
    monitor = InputMonitor(
        "shift_l",
        "alt+up",
        "alt+down",
        lambda: calls.append("previous"),
        lambda: calls.append("next"),
    )
    assert monitor.activation_key == "shift"
    monitor._press(Key("alt_l"))
    monitor._press(Key("up"))
    assert calls == ["previous"]
    monitor._release(Key("up"))
    monitor._press(Key("down"))
    assert calls == ["previous", "next"]
