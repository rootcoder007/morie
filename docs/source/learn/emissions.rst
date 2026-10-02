Measure the compute cost, and prove it
======================================

Every pipeline run can report what it cost in energy and CO2, and seal
that report so a reader can check it was not edited. The tracker follows
the CodeCarbon methodology (the formulas and their sources are in
:doc:`../methods/compute-emissions`); the seal is a bricklayer capsule.

Try it on your machine
----------------------

.. code-block:: bash

   morie emissions --seconds 5 --country CAN
   rmorie emissions --seconds 5 --country CAN

::

   Emissions:         1.76e-06 kg CO2eq over 5.0 s
   Energy:            1.03e-05 kWh (CPU 4.77e-06, RAM 5.56e-06)
   CPU:               Intel(R) Core(TM) i7-1185G7, TDP-scaled 8.6 W at 25.1% utilisation (machine mode)
   Carbon intensity:  0.17 kg/kWh (CAN)
   Capsule:           emissions/emissions_manifest.json + signed emissions/capsule_bundle.json

``--country`` picks the grid (ISO-3); without it the packages use
``MORIE_COUNTRY_ISO``, then a geolocation lookup, then the world average,
and the report says which. ``MORIE_EMISSIONS_OFFLINE=1`` skips the lookup.

Track a pipeline run
--------------------

.. code-block:: bash

   morie pipeline --all -y --output-dir out/        # ends with: Pipeline CO2 emissions: 0.000123 kg CO2eq
   rmorie pipeline --all --output-dir out/          # same, from R; --no-carbon turns it off

Both write ``out/emissions/emissions.csv`` (one row per run, the CodeCarbon
column layout, so rows from both languages can be concatenated) and the
capsule next to it.

From code
---------

.. code-block:: python

   from morie.emissions import EmissionsTracker
   from morie.modules import run_module

   def run_everything():
       run_module("power-design", output_dir="out/power-design")

   with EmissionsTracker(project_name="my-analysis", output_dir="out/emissions") as t:
       run_everything()
   print(t.capsule)        # {'manifest': ..., 'bundle': ..., 'signed': True/False}

.. code-block:: r

   run_everything <- function() rmorie::morie_run_morie_module("power-design", output_dir = "out/power-design")
   r <- rmorie::morie_emissions_track(run_everything(), project_name = "my-analysis",
                                      output_dir = "out/emissions")
   r$emissions_kg; r$capsule
   rmorie::morie_emissions_verify("out/emissions")$ok

The capsule
-----------

``emissions_manifest.json`` is a bricklayer manifest: the measurements,
the method and its sources, the environment (interpreter, platform,
packages). ``capsule_bundle.json`` holds the SHA-256 of the CSV and the
manifest under an ML-DSA-44 signature; ``rmorie::morie_emissions_verify()``
or ``rmoriebricklayer::capsule_bundle_verify()`` re-hashes the files and
checks it, so a changed byte in either file fails. The Python package signs
through R when ``Rscript`` with ``rmoriebricklayer`` is installed and
otherwise writes the manifest only and says "unsigned".

The signing key is generated per run unless you pass your own
(``key = rmoriebricklayer::fips_keygen()`` in R), so by default the capsule
proves the files are unchanged since sealing, not who sealed them.
