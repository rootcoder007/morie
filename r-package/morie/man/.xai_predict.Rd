% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.xai_predict}
\alias{.xai_predict}
\title{.xai_predict}
\usage{
.xai_predict(predict_fn, X)
}
\arguments{
\item{predict_fn}{See Usage.}
\item{X}{A matrix; passed to \code{nrow}.}
}
\description{
A step of the fairness_xai implementation. Called by \code{morie_fairness_xai_ale}, \code{morie_fairness_xai_ceteris_paribus}, \code{morie_fairness_xai_partial_dependence} and 2 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
