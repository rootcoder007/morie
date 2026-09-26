"""nmwnp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmwnp import wnominate_prob


def test_nmwnp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wnominate_prob(data=None)
