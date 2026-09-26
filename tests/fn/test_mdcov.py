"""mdcov is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mdcov import mdcov


def test_mdcov_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mdcov()
