"""mollwd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mollwd import mollwd


def test_mollwd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mollwd()
