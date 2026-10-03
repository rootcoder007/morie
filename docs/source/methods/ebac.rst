eBAC — Estimated Blood Alcohol Concentration
=============================================

Part of :doc:`index` — MORIE's statistical-methods reference.

eBAC is a continuous outcome derived from self-reported alcohol consumption
data in CPADS. MORIE computes two eBAC variants.

Widmark formula
---------------

MORIE uses Widmark's formula in US units (Matthews & Miller, 1979):

.. math::

   \text{eBAC} = \frac{A \times 5.14}{W \times r} - 0.015\, t

where

- :math:`A` = fluid ounces of ethanol consumed; a standard drink (14 g of
  ethanol) is 0.6 fl oz, so :math:`A = 0.6 \times` drinks
- :math:`W` = body weight (lb)
- :math:`r` = Widmark distribution factor (0.73 for men, 0.66 for women)
- :math:`0.015` = elimination rate (percent BAC per hour)
- :math:`t` = hours since drinking began

The result is a percentage (g/dL); values below zero are floored at zero.
Five drinks for a 150 lb man over two hours give
:math:`5 \times 0.6 \times 5.14 / (150 \times 0.73) - 0.03 = 0.111`.

MORIE variants
-------------

``ebac_tot``
   Total eBAC from the full CPADS drinking episode as reported.

``ebac_legal``
   Binary indicator: :math:`\mathbb{1}[\text{eBAC} \geq 0.08\text{ g/dL}]`
   (Canadian legal driving limit).

Both are available as canonical CPADS variables and participate in the
``CPADS_REQUIRED_VARIABLES`` contract.

Python API
----------

.. code-block:: python

   from morie import calculate_ebac, is_over_legal_limit

   # weight in pounds; the Widmark constant is 0.73 for men, 0.66 for women
   ebac = calculate_ebac(drinks=5, weight_lbs=154, hours=2.0, gender_constant=0.73)
   over = is_over_legal_limit(ebac)

eBAC-IPW module
---------------

The ``ebac-selection-adjustment-ipw`` module uses eBAC strata as a
selection-correction mechanism. See :doc:`causal` for the statistical
framework.

References
----------

- Matthews DB, Miller WR (1979). Estimating blood alcohol concentration: two
  computer programs and their applications in therapy and research.
  *Addictive Behaviors* 4(1):55-60.
- Widmark EMP (1932). *Die theoretischen Grundlagen und die praktische
  Verwendbarkeit der gerichtlich-medizinischen Alkoholbestimmung*.
  Urban & Schwarzenberg.
- Brick J (2006). Standardization of alcohol calculations in research.
  *Alcoholism: Clinical and Experimental Research*, 30(8):1276–1287.
  https://doi.org/10.1111/j.1530-0277.2006.00155.x
