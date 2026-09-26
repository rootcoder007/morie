"""socec is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.socec import socec


def test_socec_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        socec()
