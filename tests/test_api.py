import pytest


@pytest.mark.xfail(reason="API not implemented yet")
def test_api_exists():
    import project99.api  # noqa: F401

