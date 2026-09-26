"""geinq is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.geinq import geinq


def test_geinq_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        geinq()
