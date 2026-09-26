"""zsfft is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsfft import spectral_sim


def test_zsfft_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spectral_sim(data=None)
