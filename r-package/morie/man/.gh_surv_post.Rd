% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gh_surv_post}
\alias{.gh_surv_post}
\title{.gh_surv_post}
\usage{
.gh_surv_post(t, ev, c, lam0)
}
\arguments{
\item{t}{A vector; its length is taken.}
\item{ev}{See Usage.}
\item{c}{Numeric; combined arithmetically in the body.}
\item{lam0}{Optional; may be \code{NULL}. Numeric; combined arithmetically in the body.}
}
\description{
A step of the helpers_ghosal_bnp implementation. Called by \code{morie_ghosal_neutral_right}, \code{morie_ghosal_survival_beta_process}. See the file header for the source the module follows.
}
\value{
A list with \code{times}, \code{S}, \code{H}, \code{dH}, \code{lam0}.
}
\keyword{internal}
\examples{
.gh_surv_post(t = c(1, 2, 3, 4, 5, 6, 7, 8), ev = c(1, 2, 3, 4, 5, 6, 7, 8), c = c(1, 2, 3, 4, 5, 6, 7, 8), lam0 = c(1, 2, 3, 4, 5, 6, 7, 8))
}
