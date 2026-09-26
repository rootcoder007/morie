"""nbcog is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbcog import nbcog


def test_nbcog_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbcog()
