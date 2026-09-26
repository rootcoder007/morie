"""gafdir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gafdir import gafdir


def test_gafdir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gafdir()
