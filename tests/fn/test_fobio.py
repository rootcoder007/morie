"""fobio is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fobio import fobio


def test_fobio_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fobio()
