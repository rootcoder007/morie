"""gchad is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gchad import gchad


def test_gchad_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gchad()
