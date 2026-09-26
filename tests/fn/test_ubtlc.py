"""ubtlc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubtlc import ubtlc


def test_ubtlc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubtlc()
