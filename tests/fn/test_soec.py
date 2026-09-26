"""soec is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.soec import soec


def test_soec_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        soec()
