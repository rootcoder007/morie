% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_llm_call}
\alias{.siu_llm_call}
\title{Internal: try `model` in order. The first one whose env var is set}
\usage{
.siu_llm_call(model, prompt, timeout_s = .siu_llm_default_timeout(), mock_response_text = NULL)
}
\arguments{
\item{model}{See Usage.}
\item{prompt}{See Usage.}
\item{timeout_s}{Defaults to \code{.siu_llm_default_timeout()}.}
\item{mock_response_text}{Defaults to \code{NULL}.}
}
\description{
AND whose request returns without erroring wins. `model` may be a character vector for failover (e.g. c("gemini", "ollama")). The `mock_response_text` arg exists ONLY so unit tests can exercise the surrounding R glue without hitting the network.
}
\value{
Nothing; this branch always raises.
}
\keyword{internal}
