from paper_prototype.main import run_smoke


def test_smoke() -> None:
    assert run_smoke() == "paper-prototype-dev ready"
