"""sghist is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sghist import sghist


def test_sghist_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sghist()
