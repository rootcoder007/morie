"""bayrhat is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bayrhat import r_hat


def test_bayrhat_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        r_hat(chains=None)
