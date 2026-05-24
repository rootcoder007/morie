# CRAN-conservative test cache override.
#
# Per CRAN repository policy, packages must not write to the user's
# home directory or any persistent location during R CMD check. morie's
# production cache resolves via `tools::R_user_dir("morie", "cache")`
# (correct for end users), but during tests we redirect every cache
# write to a per-session tempdir so check leaves nothing behind on the
# CRAN test machines.
#
# MORIE_CACHE_DIR env var is the single override read by
# morie_cache_dir() (see R/database.R). Setting it here puts every
# morie cache, db, and download under a fresh tempfile path that R
# auto-cleans at session exit.
Sys.setenv(MORIE_CACHE_DIR = tempfile("morie-test-cache-"))
