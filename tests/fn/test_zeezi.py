"""zeezi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zeezi import ecological_zip


def test_zeezi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ecological_zip(data=None)
