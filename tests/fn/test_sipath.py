"""sipath is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sipath import sipath


def test_sipath_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sipath()
