"""aitvarc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aitvarc import aitchison_clr_covariance


def test_aitvarc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aitchison_clr_covariance(X=None)
