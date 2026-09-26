"""saamo is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.saamo import saamo


def test_saamo_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        saamo()
