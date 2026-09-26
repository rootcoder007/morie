"""xrsem is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrsem import sem_ml


def test_xrsem_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sem_ml(data=None)
