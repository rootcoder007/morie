"""fourT is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fourT import fourier_transform


def test_fourT_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fourier_transform(f=None, x=None, k=None)
