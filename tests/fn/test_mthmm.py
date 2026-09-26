"""mthmm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mthmm import mthmm


def test_mthmm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mthmm()
