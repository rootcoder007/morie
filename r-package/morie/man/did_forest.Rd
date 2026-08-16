% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{did_forest}
\alias{did_forest}
\title{Heterogeneous ATT: DiD taken locally under forest weights}
\usage{
did_forest(Y, D, X, event_time, x_eval = NULL, n_trees = 200L, min_leaf = 5L, alpha = 0.05, max_depth = 12L, seed = 0L, kind = "double-sample", clusters = NULL)
}
\arguments{
\item{Y}{Balanced n-by-T panel.}
\item{D}{0/1 adoption vector.}
\item{X}{Covariate matrix.}
\item{event_time}{\code{H}.}
\item{x_eval}{Optional points at which to evaluate; default   training rows.}
\item{n_trees}{Number of trees.}
\item{min_leaf}{Minimum leaf size.}
\item{alpha}{Honest subsample fraction.}
\item{max_depth}{Maximum tree depth.}
\item{seed}{Seed for the shared generator.}
\item{kind}{Forest subsampling scheme passed to \code{hntfst}.}
\item{clusters}{Optional cluster IDs.}
}
\description{
Heterogeneous ATT: DiD taken locally under forest weights
}
\value{
A list mirroring the Python \code{RichResult} payload.
}
\keyword{internal}
