"""svcmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svcmp import committee_med


def test_svcmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        committee_med(data=None)
