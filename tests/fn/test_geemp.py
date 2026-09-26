"""geemp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.geemp import geemp


def test_geemp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        geemp()
