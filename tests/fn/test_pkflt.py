"""pkflt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pkflt import pkflt


def test_pkflt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pkflt()
