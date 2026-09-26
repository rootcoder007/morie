"""mvtfe is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mvtfe import mvtfe


def test_mvtfe_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mvtfe()
