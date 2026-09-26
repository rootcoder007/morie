"""nbcnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbcnt import nbcnt


def test_nbcnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbcnt()
