"""srml is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srml import srml


def test_srml_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srml()
