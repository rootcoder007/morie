"""seclt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seclt import seclt


def test_seclt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seclt()
