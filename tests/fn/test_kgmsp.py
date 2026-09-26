"""kgmsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgmsp import kriging_mspe


def test_kgmsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kriging_mspe(values=None, x=None)
