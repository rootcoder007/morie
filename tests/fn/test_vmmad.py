"""vmmad is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmmad import vmmad


def test_vmmad_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmmad()
