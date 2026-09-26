"""ismul is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ismul import ismul


def test_ismul_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ismul()
