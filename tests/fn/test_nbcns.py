"""nbcns is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbcns import nbcns


def test_nbcns_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbcns()
