"""prsval is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.prsval import presmessick_validity


def test_prsval_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        presmessick_validity(evidence_set=None)
