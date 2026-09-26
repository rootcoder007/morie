"""telmt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.telmt import telemetry_drift


def test_telmt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        telemetry_drift(error_stream=None)
