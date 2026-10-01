Compute emissions
=================

``morie.emissions`` (Python) and ``morie_emissions_*`` (R) estimate the
energy and CO2-equivalent of a computation the way CodeCarbon does, without
its dependencies. The usage page is :doc:`../learn/emissions`.

Energy
------

Power is sampled every ``measure_power_secs`` and integrated:

.. math::

   E = \sum_i \left( P_{cpu}(u_i) + P_{ram} \right) \Delta t_i,
   \qquad P_{cpu}(u) = TDP \left(0.1 + 0.9\,u^3\right)

where :math:`u_i` is the machine's CPU utilisation over the sample (the
kernel's aggregate counters: ``/proc/stat`` on Linux, ``host_statistics``
on macOS; the Python arm reads them from a daemon thread, the R arm from a
C++ ``std::thread``, so the same cubic weighting applies per sample in both)
and TDP is looked up from the CPU model name (Apple M-series 10-15 W, Core
i5/i7 65 W, i9 125 W, Ryzen 7 65 W, Ryzen 9 105 W, Xeon 150 W, otherwise
85 W). RAM power is 5 W per DIMM on x86 and 1.5 W on ARM with the
CodeCarbon DIMM-count estimate from total memory and its efficiency scaling
beyond four DIMMs (minimum 10 W / 3 W). No GPU power is counted. On a
platform without CPU counters the R arm uses the process's CPU time over
wall time and records ``tracking_mode = process``.

Emissions
---------

.. math::

   CO_2eq = E \times PUE \times I_{grid}, \qquad H_2O = E \times PUE \times WUE

:math:`I_{grid}` (kg CO2eq per kWh) comes from the IEA / Our World in Data
energy mix shipped with both packages (213 countries): the country's
published intensity when it has one, otherwise the generation-weighted
mean of the per-source intensities (coal 995, petroleum 816, natural gas
743, fossil 635, geothermal 38, hydro 26, nuclear 29, solar 48, wind 26
gCO2/kWh), and the world average of 475 g/kWh for an unknown country. The
country is the ``country_iso_code`` argument, else ``MORIE_COUNTRY_ISO``,
else a geolocation lookup (skipped under ``MORIE_EMISSIONS_OFFLINE``).

The CSV row
-----------

One row per run in the CodeCarbon layout (36 columns: timestamp, project,
run id, duration, emissions and rate, CPU / GPU / RAM power and energy,
total energy, water, country, OS, interpreter, CPU count and model, RAM
size, tracking mode, utilisation, PUE, WUE, ...), so rows from either
package or from CodeCarbon itself concatenate.

Provenance
----------

Each run is sealed in a bricklayer capsule: ``emissions_manifest.json``
(measurements, method, sources, environment) and ``capsule_bundle.json``
(SHA-256 of the CSV and the manifest under an ML-DSA-44 signature;
``rmoriebricklayer::capsule_bundle()``). ``rmorie::morie_emissions_verify()``
checks it. The key is per run unless supplied, which makes the capsule a
proof of integrity, not of authorship.

Limits
------

The TDP-times-utilisation model is an estimate, not a measurement: it
ignores GPUs, frequency scaling and the rest of the machine, and the RAM
heuristic is coarse. A machine with a power meter or RAPL counters will
disagree with it by a factor that depends on the workload. The figure is
comparable across runs on one machine and across the two packages, which
is what the capsule is for.

References
----------

- CodeCarbon, https://github.com/mlco2/codecarbon (methodology and the
  energy-mix data, MIT licence).
- IEA, *Global Energy and CO2 Status Report*; Our World in Data,
  electricity mix by country.
