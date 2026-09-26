"""pdesl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pdesl import pde_separation


def test_pdesl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pde_separation(pde=None)
