"""soril is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.soril import soril


def test_soril_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        soril()
