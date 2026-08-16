% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_llm_freeapi_completion}
\alias{.morie_llm_freeapi_completion}
\title{Non-streaming FreeAPI completion via the same OpenAI-compatible}
\usage{
.morie_llm_freeapi_completion(messages, model = NULL, timeout = 180)
}
\arguments{
\item{messages}{Carried through into a list the body builds.}
\item{model}{Defaults to \code{NULL}.}
\item{timeout}{Defaults to \code{180}.}
}
\description{
endpoint that ollama exposes.  R has no native ``ollamafreeapi`` SDK, so we POST /v1/chat/completions to FREEAPI_BASE_URL.  Returns "" on failure.
}
\value{
The value of \code{.morie_llm_strip_think}.
}
\keyword{internal}
