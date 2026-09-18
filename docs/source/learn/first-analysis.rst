Your first analysis, end-to-end
================================

We will answer a real question with a real-shaped dataset, in a dozen
lines of code, with proper margins of error.

The question
------------

MORIE ships a 1,200-row frame with the schema of the Canadian
Postsecondary Alcohol and Drug Use Survey (CPADS): survey weight,
past-year alcohol use, heavy drinking in the last 30 days, estimated
blood-alcohol content (``ebac_tot``), cannabis use, age group, gender,
region, and self-rated mental and physical health. The values are
synthetic (the real PUMF is loaded instead when it is present on disk,
and a warning tells you which one you got).

**Question:** *Do students who used cannabis have a different estimated
blood-alcohol content than students who did not, and by how much?*

This is a difference-of-means question. Two groups, one numeric
outcome, you want a number with error bars.

----

Step 1: Load the data
---------------------

.. code-block:: python

   from morie import datasets

   df = datasets.cpads()          # bundled synthetic frame, or the real PUMF if present
   print(len(df), list(df.columns))
   # 1200 ['weight', 'alcohol_past12m', 'heavy_drinking_30d', 'ebac_tot', ...]

``df`` is MORIE's native DataFrame: no pandas or NumPy is installed or
needed. It indexes like the frames you know (``df["ebac_tot"]``,
``df.loc[...]``) and any function in the package accepts it. If you
already have a pandas frame, pass it as it is; it is converted on entry.

If you've never seen ``import``, ``=``, or ``from``, the
`Python tutorial <https://docs.python.org/3/tutorial/>`_ has a
one-page introduction. Five minutes is enough.

----

Step 2: Pick the right tool
---------------------------

The question is "are the means different between two groups?". That
maps to a *two-sample t-test*, which lives at
``morie.fn.two_sample_t_test`` (Welch's version by default, so unequal
variances are fine).

You can find this by:

- :doc:`../methods/inference_engine`, the methods reference.
- The catalogue of every ``morie.fn`` callable: :doc:`../api/fn-catalog`.
- ``help(morie.fn.two_sample_t_test)`` at the Python prompt, or
  ``morie cheatsheet`` at the terminal.

----

Step 3: Run the test
--------------------

.. code-block:: python

   from morie.fn import two_sample_t_test

   ebac = df["ebac_tot"]
   used = df["cannabis_any_use"]
   users     = [e for e, u in zip(ebac, used) if u == 1]
   non_users = [e for e, u in zip(ebac, used) if u == 0]

   result = two_sample_t_test(users, non_users, alternative="two-sided")
   print(result)
   # {'t': ..., 'df': ..., 'p_value': ..., 'mean_diff': ...,
   #  'ci_diff_lower': ..., 'ci_diff_upper': ..., 'method': 'Welch two-sample t-test'}

The keys are:

- ``mean_diff``, the difference of means (users minus non-users)
- ``t`` and ``df``, the t-statistic and Welch degrees of freedom
- ``p_value``
- ``ci_diff_lower``, ``ci_diff_upper``, the 95% confidence interval

----

Step 4: Read the output honestly
--------------------------------

Three numbers matter:

1. **The estimate.**  How big is the difference?  eBAC values sit
   around 0 to 0.15, so a difference of 0.04 would be large and a
   difference of 0.0004 is nothing, even if the p-value were tiny.

2. **The confidence interval.**  Where could the truth plausibly
   be, given this much data?  An interval of (0.03, 0.05) means
   "I'm pretty sure the real difference is somewhere in there".
   An interval of (-0.008, 0.007) that straddles zero means
   "the data cannot tell these two groups apart".

3. **The p-value.**  How surprising would the data be if the truth
   really is "no difference"?  A p of 0.001 means "very surprising,
   probably a real difference".  A p of 0.9 means "not surprising at
   all".  **The p-value is not the probability the difference is
   real**; it is the probability of the data given no difference.
   Those are different statements.

On the synthetic frame the answer is a mean difference near zero with
an interval straddling zero: the fake values carry no cannabis effect,
which is exactly what the warning at load time is telling you.

----

Step 5: What to do when the data is messy
-----------------------------------------

The t-test assumes both groups are roughly normal.  Real data isn't.
For income, time-on-page, hospital cost, whatever "engagement" means in
your A/B test, it usually isn't.

When the data is skewed or has extreme values, reach for the
rank-based and robust alternatives (``morie.fn.mann_whitney_test``,
``morie.fn.wilcoxon_signed_rank_test``, the robust regression family).
``help()`` on any of them tells you when to use it.

----

Where to go next
----------------

- :doc:`../methods/index`, the full statistical-methods reference,
  sorted by question. Start here for the catalogue of estimators
  and which one fits which design.
- :doc:`../methods/causal`, once your question is causal ("did X
  cause Y?") and not just correlational. The same frame works there:
  ``morie.causal.estimate_double_ml(df, outcome="heavy_drinking_30d",
  treatment="cannabis_any_use", covariates=["age_group", "gender"])``.

----

The big idea, restated
----------------------

You loaded data, picked a tool that matched the question, ran it,
and got a number with a margin of error.  That is the entire workflow.
The eighteen thousand other callables in MORIE exist because the
question can take eighteen thousand other shapes, but the shape of the
work doesn't change.
