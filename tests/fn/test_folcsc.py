"""folcsc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.folcsc import folcsc


def test_folcsc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        folcsc()
