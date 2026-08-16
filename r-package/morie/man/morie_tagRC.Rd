% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_tagRC}
\alias{morie_tagRC}
\title{morie_tagRC}
\usage{
morie_tagRC(triples, focus, d = 0.7, weight = 0.9, iters = 200)
}
\arguments{
\item{triples}{Passed to \code{tagRC_tripartite_graph}.}
\item{focus}{Passed to \code{tagRC_preference_vector}.}
\item{d}{Passed to \code{tagRC_adapted_pagerank}. Defaults to \code{0.7}.}
\item{weight}{Passed to \code{tagRC_preference_vector}. Defaults to \code{0.9}.}
\item{iters}{Passed to \code{tagRC_adapted_pagerank}. Defaults to \code{200}.}
}
\description{
Part of the tagRC_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{ranking}, \code{difference}, \code{with_preference}, \code{without_preference}, \code{undifferenced_ranking}, \code{baseline_ranking}, \code{focus}, \code{n_nodes}, \code{method}, \code{note}.
}
\keyword{internal}
