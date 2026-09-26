"""vgani is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgani import anisotropy_ratio


def test_vgani_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        anisotropy_ratio(data=None)
