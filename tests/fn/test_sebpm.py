"""sebpm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sebpm import sebpm


def test_sebpm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sebpm()
