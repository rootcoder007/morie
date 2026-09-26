"""zsrch is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsrch import random_chisq_field


def test_zsrch_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        random_chisq_field(data=None)
