"""cshts is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cshts import cshts


def test_cshts_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cshts()
