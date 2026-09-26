"""ksmsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ksmsp import ksmsp


def test_ksmsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ksmsp()
