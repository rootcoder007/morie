"""seprf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seprf import seprf


def test_seprf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seprf()
