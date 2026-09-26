"""crssmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.crssmp import crssmp


def test_crssmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        crssmp()
