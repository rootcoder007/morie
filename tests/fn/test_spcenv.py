"""spcenv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spcenv import spcenv


def test_spcenv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spcenv()
