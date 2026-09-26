"""nbrwy is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbrwy import nbrwy


def test_nbrwy_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbrwy()
