"""ptisg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptisg import isotropic_guard


def test_ptisg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        isotropic_guard(data=None)
