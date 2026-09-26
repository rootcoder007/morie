"""aglsi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aglsi import aglsi


def test_aglsi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aglsi()
