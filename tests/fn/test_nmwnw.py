"""nmwnw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmwnw import wnominate_weight


def test_nmwnw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wnominate_weight(data=None)
