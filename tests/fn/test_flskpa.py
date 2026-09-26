"""flskpa is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.flskpa import fleiss_kappa


def test_flskpa_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fleiss_kappa(X=None)
