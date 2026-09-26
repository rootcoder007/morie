"""balsmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.balsmp import balsmp


def test_balsmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        balsmp()
