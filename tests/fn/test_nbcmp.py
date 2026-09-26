"""nbcmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbcmp import nbcmp


def test_nbcmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbcmp()
