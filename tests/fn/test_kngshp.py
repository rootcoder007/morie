"""kngshp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kngshp import kinship_estimator


def test_kngshp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kinship_estimator(genotypes=None)
