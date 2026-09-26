"""sgcosm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgcosm import sgcosm


def test_sgcosm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgcosm()
