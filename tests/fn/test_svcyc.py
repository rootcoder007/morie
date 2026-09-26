"""svcyc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svcyc import condorcet_cycle


def test_svcyc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        condorcet_cycle(data=None)
