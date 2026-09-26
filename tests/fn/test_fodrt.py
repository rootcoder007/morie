"""fodrt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fodrt import fodrt


def test_fodrt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fodrt()
