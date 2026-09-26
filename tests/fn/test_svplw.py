"""svplw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svplw import polarization_wolf


def test_svplw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        polarization_wolf(data=None)
