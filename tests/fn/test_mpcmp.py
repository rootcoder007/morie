"""mpcmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mpcmp import mpcmp


def test_mpcmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mpcmp()
