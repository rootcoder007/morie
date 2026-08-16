% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_llm_messages}
\alias{.morie_llm_messages}
\title{.morie_llm_messages}
\usage{
.morie_llm_messages(prompt, context = NULL, system_prompt = NULL)
}
\arguments{
\item{prompt}{Carried through into a list the body builds.}
\item{context}{Optional; may be \code{NULL}. A vector; indexed elementwise.}
\item{system_prompt}{Optional; may be \code{NULL}. Carried through into a list the body builds.}
}
\description{
Part of the llm implementation; see the file header for the source it follows.
}
\value{
The value of \code{list}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.morie_llm_messages(V)
}
