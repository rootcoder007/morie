// SPDX-License-Identifier: AGPL-3.0-or-later
//
// libmorie -- Hawkes-process likelihood kernels (registration).
#pragma once

#include <nanobind/nanobind.h>

// Binds the Hawkes-process kernels into the morie._core module.
void register_hawkes(nanobind::module_ &m);
