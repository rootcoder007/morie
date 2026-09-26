"""pshmm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pshmm import pshmm


def test_pshmm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pshmm()
