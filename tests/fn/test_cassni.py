"""cassni is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cassni import cassni


def test_cassni_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cassni()
