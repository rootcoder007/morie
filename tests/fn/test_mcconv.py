"""mcconv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mcconv import mcconv


def test_mcconv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mcconv()
