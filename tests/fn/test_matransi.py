"""matransi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.matransi import ma_logit_inverse


def test_matransi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ma_logit_inverse(z=None)
