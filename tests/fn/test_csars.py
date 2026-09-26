"""csars is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csars import csars


def test_csars_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csars()
