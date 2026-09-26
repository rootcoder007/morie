"""srsxe is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srsxe import srsxe


def test_srsxe_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srsxe()
