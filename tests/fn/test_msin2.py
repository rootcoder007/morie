"""msin2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msin2 import indscal_weights


def test_msin2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        indscal_weights(data=None)
