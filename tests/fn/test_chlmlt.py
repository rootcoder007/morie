"""chlmlt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chlmlt import chlmlt


def test_chlmlt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chlmlt()
