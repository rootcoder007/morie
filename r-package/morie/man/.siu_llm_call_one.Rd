% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_llm_call_one}
\alias{.siu_llm_call_one}
\title{Default timeout is 600s (10 min) -- long enough to accommodate}
\usage{
.siu_llm_call_one(model, prompt, timeout_s = .siu_llm_default_timeout())
}
\arguments{
\item{model}{See Usage.}
\item{prompt}{Passed to \code{morie_llm_ask}.}
\item{timeout_s}{Defaults to \code{.siu_llm_default_timeout()}.}
}
\description{
slow CPU-only local Ollama generation on a Raspberry Pi. Override via MORIE_LLM_TIMEOUT_S env var or the timeout_s arg.
}
\value{
The value of \code{p$extract}.
}
\keyword{internal}
