"""reldge is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.reldge import reliability_gebv


def test_reldge_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        reliability_gebv(fit=None)
