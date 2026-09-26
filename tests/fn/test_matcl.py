"""matcl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.matcl import matcl


def test_matcl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        matcl()
