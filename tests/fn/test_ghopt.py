"""ghopt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghopt import ghopt


def test_ghopt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghopt()
