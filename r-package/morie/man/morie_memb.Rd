% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_memb}
\alias{morie_memb}
\title{morie_memb}
\usage{
morie_memb(target_predict, shadow_data, eval_in, eval_out, train_fn = NULL, attack_train_fn = NULL, n_shadow = NULL, sort_features = FALSE, threshold = 0.5)
}
\arguments{
\item{target_predict}{Passed to \code{memb}.}
\item{shadow_data}{Passed to \code{memb}.}
\item{eval_in}{Passed to \code{memb}.}
\item{eval_out}{Passed to \code{memb}.}
\item{train_fn}{Passed to \code{memb}.}
\item{attack_train_fn}{Passed to \code{memb}.}
\item{n_shadow}{Passed to \code{memb}.}
\item{sort_features}{Passed to \code{memb}. Defaults to \code{FALSE}.}
\item{threshold}{Passed to \code{memb}. Defaults to \code{0.5}.}
}
\description{
Part of the memb_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{memb}.
}
\keyword{internal}
