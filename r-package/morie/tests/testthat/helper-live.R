# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Test helper: run a live-network expression but SKIP (never fail) the test
# if it is unavailable — offline, API change, HTML-instead-of-JSON parse
# error, timeout, etc. Also skips on CRAN, since live network access in
# tests is against CRAN policy. Use this to wrap any call that hits a real
# external endpoint.
skip_if_live_unavailable <- function(expr) {
  testthat::skip_on_cran()
  tryCatch(
    force(expr),
    error = function(e)
      testthat::skip(paste0("live API unavailable: ", conditionMessage(e)))
  )
}
