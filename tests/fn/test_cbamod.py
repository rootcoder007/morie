"""cbamod is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cbamod import cbam_attention


def test_cbamod_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cbam_attention(x=None)
