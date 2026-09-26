"""ptrng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptrng import pp_intensity


def test_ptrng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pp_intensity(data=None)
