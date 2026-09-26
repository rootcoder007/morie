"""zemir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zemir import migration_flow


def test_zemir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        migration_flow(data=None)
