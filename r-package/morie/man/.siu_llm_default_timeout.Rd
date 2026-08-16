% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_llm_default_timeout}
\alias{.siu_llm_default_timeout}
\title{Providers:}
\usage{
.siu_llm_default_timeout()
}
\description{
gemini -- closed, paid, fast.  env: GOOGLE_API_KEY claude -- closed, paid, fast.  env: ANTHROPIC_API_KEY ollama -- open-weight models env: OLLAMA_HOST (e.g. over a local or self- "http://localhost:11434" hosted REST endpoint; or any hosted Ollama- free/OllamaFreeAPI- compatible base URL), compatible.  optional OLLAMA_MODEL (default "llama3.2:3b") Internal: default LLM HTTP timeout in seconds. 600s (10 min) accommodates slow CPU-only local inference on a Raspberry Pi. Override globally via MORIE_LLM_TIMEOUT_S env var.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
