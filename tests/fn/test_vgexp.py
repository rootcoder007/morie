"""vgexp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgexp import vario_exponential


def test_vgexp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_exponential(coords=None, values=None)
