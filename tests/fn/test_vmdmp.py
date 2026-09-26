"""vmdmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmdmp import vmdmp


def test_vmdmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmdmp()
