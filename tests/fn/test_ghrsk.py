"""ghrsk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghrsk import ghrsk


def test_ghrsk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghrsk()
