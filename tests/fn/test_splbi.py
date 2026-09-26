"""splbi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.splbi import splbi


def test_splbi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        splbi()
