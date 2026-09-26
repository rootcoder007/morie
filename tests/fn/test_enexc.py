"""enexc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enexc import enexc


def test_enexc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enexc()
