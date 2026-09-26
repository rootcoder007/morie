"""lambrt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lambrt import lambrt


def test_lambrt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lambrt()
