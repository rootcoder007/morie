"""mcvar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mcvar import mcvar


def test_mcvar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mcvar()
