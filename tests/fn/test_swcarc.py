"""swcarc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swcarc import swcarc


def test_swcarc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swcarc(W=None)
