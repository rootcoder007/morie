"""ptlgc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptlgc import log_gaussian_cox


def test_ptlgc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        log_gaussian_cox(data=None)
