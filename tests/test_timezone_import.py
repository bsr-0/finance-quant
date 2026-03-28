import pipeline.snapshot.contract_snapshots as cs


def test_timezone_available():
    assert hasattr(cs, "UTC")
