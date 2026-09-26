"""svlss is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svlss import loss_function


def test_svlss_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        loss_function(data=None)
