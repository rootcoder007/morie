"""sicosm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sicosm import sicosm


def test_sicosm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sicosm()
