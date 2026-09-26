"""vggau is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vggau import vario_gaussian


def test_vggau_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_gaussian(coords=None, values=None)
