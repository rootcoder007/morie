"""gegni is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gegni import gegni


def test_gegni_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gegni()
