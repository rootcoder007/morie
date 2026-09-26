"""ptkda is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptkda import kde_adaptive


def test_ptkda_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kde_adaptive(data=None)
