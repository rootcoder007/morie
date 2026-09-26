"""ppgey is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppgey import ppgey


def test_ppgey_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppgey()
