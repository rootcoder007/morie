"""sigpep is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sigpep import signal_peptide


def test_sigpep_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        signal_peptide(sequence=None)
