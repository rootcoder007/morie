"""fosnwl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fosnwl import fosnwl


def test_fosnwl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fosnwl()
