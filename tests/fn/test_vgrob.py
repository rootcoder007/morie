"""vgrob is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgrob import vario_robust


def test_vgrob_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_robust(coords=None, values=None)
