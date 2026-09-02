% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{cgnonl_line_search_fr}
\alias{cgnonl_line_search_fr}
\title{cgnonl_line_search_fr}
\usage{
cgnonl_line_search_fr(f, grad, x, p, f0, g0, est = NULL, max_double = 60L, max_cubic = 40L, 
    tol = 1e-12) 
}
\arguments{
\item{f}{Accepted by the signature and not used anywhere in the body.}
\item{grad}{Accepted by the signature and not used anywhere in the body.}
\item{x}{A vector; its length is taken.}
\item{p}{Numeric; combined arithmetically in the body.}
\item{f0}{Numeric; combined arithmetically in the body.}
\item{g0}{Passed to \code{.cgnonl_dot}.}
\item{est}{Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.}
\item{max_double}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{60L}.}
\item{max_cubic}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{40L}.}
\item{tol}{Numeric; combined arithmetically in the body. Defaults to \code{1e-12}.}
}
\description{
A step of the cgnonl_native implementation. Called by \code{cgnonl_nonlinear_cg}. See the file header for the source the module follows.
}
\value{
A list with \code{t}, \code{x}, \code{f}, \code{g}, \code{n_eval}.
}
\keyword{internal}
\examples{
f <- function(x) sum((x - c(1, 2))^2)
grad <- function(x) 2 * (x - c(1, 2))
x <- c(0, 0)
g0 <- grad(x)
ls <- cgnonl_line_search_fr(f, grad, x, -g0, f(x), g0)
is.list(ls) || is.numeric(ls)
}
