"""semtl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.semtl import semtl


def test_semtl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        semtl()
