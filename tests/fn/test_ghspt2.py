"""ghspt2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghspt2 import ghspt2


def test_ghspt2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghspt2()
