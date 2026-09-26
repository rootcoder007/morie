"""gafacc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gafacc import gafacc


def test_gafacc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gafacc()
