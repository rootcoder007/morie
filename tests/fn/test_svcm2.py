"""svcm2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svcm2 import committee_2d


def test_svcm2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        committee_2d(data=None)
