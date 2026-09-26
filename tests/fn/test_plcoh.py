"""plcoh is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.plcoh import plcoh


def test_plcoh_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        plcoh()
