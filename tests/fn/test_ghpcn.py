"""ghpcn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghpcn import ghpcn


def test_ghpcn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghpcn()
