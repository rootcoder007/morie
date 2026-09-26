"""dtiws is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtiws import dtiws


def test_dtiws_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtiws()
