"""vgemp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgemp import empirical_vario


def test_vgemp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        empirical_vario(coords=None, values=None)
