"""matrans is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.matrans import ma_logit_transform


def test_matrans_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ma_logit_transform(p=None, n=None)
