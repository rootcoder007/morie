"""mcqmc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mcqmc import mcqmc


def test_mcqmc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mcqmc()
