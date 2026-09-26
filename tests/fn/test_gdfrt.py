"""gdfrt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdfrt import gdfrt


def test_gdfrt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdfrt()
