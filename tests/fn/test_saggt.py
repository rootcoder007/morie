"""saggt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.saggt import saggt


def test_saggt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        saggt()
