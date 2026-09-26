"""kgqnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgqnt import kriging_quantile


def test_kgqnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kriging_quantile(values=None, x=None)
