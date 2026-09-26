"""fotrsg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fotrsg import fotrsg


def test_fotrsg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fotrsg()
