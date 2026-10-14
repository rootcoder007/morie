# Extracted from test-cli-entry.R:56

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "morie", path = "..")
attach(test_env, warn.conflicts = FALSE)

# prequel ----------------------------------------------------------------------
.pkg <- if (isNamespaceLoaded("morie")) "morie" else "morie"
.capture <- function(...) {
  buf <- character()
  status <- morie_cli(c(...), out = function(s) buf <<- c(buf, s))
  list(text = paste(buf, collapse = ""), status = status)
}

# test -------------------------------------------------------------------------
ps <- rmoriebricklayer::bricklayer_llm_providers()
withr::local_envvar(c(stats::setNames(rep(NA_character_, nrow(ps)), ps$key_env), XDG_CONFIG_HOME = withr::local_tempdir(),
                        OLLAMA_HOST = "off", MORIE_LLM_BASE_URL = NA, MORIE_HOSTED_KEY = NA))
d <- .capture("doctor")
expect_equal(d$status, 0L)
expect_match(d$text, "local Ollama +disabled")
expect_match(d$text, "ask has no route yet")
expect_match(d$text, "data key: none \\(`rmorie login`")
