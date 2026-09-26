"""tsshm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsshm import tsshm


def test_tsshm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsshm()
