"""kgckw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgckw import cok_weights


def test_kgckw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cok_weights(data=None)
