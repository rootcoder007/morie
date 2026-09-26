"""Rainbow: 6 DQN improvements combined."""

__all__ = ["rainbow_dqn"]


def rainbow_dqn(env):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Rainbow: 6 DQN improvements combined

    Formula: double + dueling + PER + multistep + noisy + categorical

    Parameters
    ----------
    env : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hessel et al (2018)
    """
    raise NotImplementedError(
        "morie.fn.raindq.rainbow_dqn is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "raindq: Rainbow: 6 DQN improvements combined"


# compact alias per ledger/NAMING.md
rainbowdqn = rainbow_dqn
