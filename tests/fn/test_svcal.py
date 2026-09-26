"""svcal is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svcal import calvert_model


def test_svcal_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        calvert_model(data=None)
