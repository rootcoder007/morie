"""entmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.entmp import entmp


def test_entmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        entmp()
