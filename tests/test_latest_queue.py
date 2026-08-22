from bidan_lens.latest_queue import LatestQueue


def test_latest_queue_discards_stale_value() -> None:
    queue = LatestQueue[int]()
    queue.put(1)
    queue.put(2)
    queue.put(3)
    assert queue.get() == 3
