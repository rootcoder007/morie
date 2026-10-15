# R/llm.R, the PR #99 line the pr77 file leaves: a probe answer already cached is returned as is,
# and no HTTP request is made. No network: the libcurl primitive is a counter.

.l9 <- function(name) get(name, envir = environment(morie_llm_ask))

test_that("a cached probe answer is returned without asking the server", {
  dir <- withr::local_tempdir()
  withr::local_envvar(OLLAMA_HOST = "http://127.0.0.1:9", OLLAMA_API_KEY = NA, XDG_CONFIG_HOME = dir,
                      "_R_CHECK_PACKAGE_NAME_" = NA)
  cache <- .l9(".morie_llm_cache")
  withr::defer(cache$ollama_cached <- NULL)
  n <- 0L
  testthat::local_mocked_bindings(
    .morie_http_get_with_status = function(url, timeout_s = 60L, headers = character(), user_agent = "",
                                           follow_redirects = TRUE) {
      n <<- n + 1L
      list(status_code = 200L, body = "{\"models\":[]}")
    },
    .package = "morie")
  cache$ollama_cached <- structure(TRUE, models = "m-one")
  expect_true(morie_llm_probe_ollama())
  cache$ollama_cached <- structure(FALSE, models = NULL)
  expect_false(morie_llm_probe_ollama())
  expect_identical(n, 0L)
  # the counter is live: with the cache cleared the same call does reach the server once
  cache$ollama_cached <- NULL
  expect_true(morie_llm_probe_ollama())
  expect_identical(n, 1L)
})
