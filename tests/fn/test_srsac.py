"""srsac is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srsac import srsac


def test_srsac_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srsac()
