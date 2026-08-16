% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.viable_terms}
\alias{.viable_terms}
\title{Keep only model terms with variation in `data`: factors need >= 2}
\usage{
.viable_terms(data, terms)
}
\arguments{
\item{data}{A vector; indexed elementwise.}
\item{terms}{A vector; indexed elementwise.}
}
\description{
observed levels, other columns >= 2 unique non-NA values. Degenerate terms (common in small or synthetic CPADS extracts) otherwise abort model.matrix with "contrasts can be applied only to factors with 2 or more levels".
}
\value{
The value of \code{[}.
}
\keyword{internal}
