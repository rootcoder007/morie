"""shorE is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.shorE import shor_factoring


def test_shorE_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        shor_factoring(N=None)
