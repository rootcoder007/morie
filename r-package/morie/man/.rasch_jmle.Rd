% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.rasch_jmle}
\alias{.rasch_jmle}
\title{Joint MAP for the Rasch model.  Difficulties are centred each cycle}
\usage{
.rasch_jmle(X, iters = 200, prior_var = 4)
}
\arguments{
\item{X}{See Usage.}
\item{iters}{Defaults to \code{200}.}
\item{prior_var}{Defaults to \code{4}.}
}
\description{
(the identification constraint), and a N(0, prior_var) penalty is carried on both sets of parameters because joint ML alone DIVERGES on perfectly separated response patterns.
}
\value{
A list with \code{b}, \code{th}.
}
\keyword{internal}
