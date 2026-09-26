"""splgtlm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.splgtlm import splgtlm


def test_splgtlm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        splgtlm(y=None, X=None, W=None)
