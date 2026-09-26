"""nbcvd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbcvd import nbcvd


def test_nbcvd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbcvd()
