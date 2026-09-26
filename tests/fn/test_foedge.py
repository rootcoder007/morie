"""foedge is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.foedge import foedge


def test_foedge_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        foedge()
