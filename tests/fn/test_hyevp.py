"""hyevp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyevp import hyevp


def test_hyevp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyevp()
