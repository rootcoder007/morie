// SPDX-License-Identifier: AGPL-3.0-or-later
// Registration hook for the linalg/elementwise/FFT kernel bindings.
#pragma once
#include <nanobind/nanobind.h>
void register_linalg(nanobind::module_ &m);
