"""semsbn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.semsbn import sem_sb_chi_sq


def test_semsbn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sem_sb_chi_sq(fit=None)
