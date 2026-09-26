"""geinc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.geinc import geinc


def test_geinc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        geinc()
