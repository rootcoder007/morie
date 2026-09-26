"""ppnnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppnnd import ppnnd


def test_ppnnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppnnd()
