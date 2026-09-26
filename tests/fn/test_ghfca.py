"""ghfca is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghfca import ghfca


def test_ghfca_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghfca()
