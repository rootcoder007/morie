% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_prxgms}
\alias{morie_prxgms}
\alias{prox_gradient}
\alias{proximal_gradient_method}
\alias{prxgms}
\title{Proximal gradient / FISTA on \code{f + g}}
\usage{
morie_prxgms(fun, grad, prox, x0, L = 1, max_iter = 500L, tol = 1e-10, accelerate = TRUE, 
    backtrack = FALSE, eta = 2, g_fun = NULL) 

prox_gradient(fun, grad, prox, x0, L = 1, max_iter = 500L, tol = 1e-10, accelerate = TRUE, 
    backtrack = FALSE, eta = 2, g_fun = NULL) 

proximal_gradient_method(fun, grad, prox, x0, L = 1, max_iter = 500L, tol = 1e-10, accelerate = TRUE, 
    backtrack = FALSE, eta = 2, g_fun = NULL) 

prxgms(fun, grad, prox, x0, L = 1, max_iter = 500L, tol = 1e-10, accelerate = TRUE, 
    backtrack = FALSE, eta = 2, g_fun = NULL) 
}
\arguments{
\item{fun}{Function \code{(x) -> float}, the smooth part \code{f}.}
\item{grad}{Function \code{(x) -> numeric}, the gradient of \code{f}.}
\item{prox}{Function \code{(x, t) -> numeric}, prox of \code{g * t}.}
\item{x0}{Initial point.}
\item{L}{Initial Lipschitz estimate.}
\item{max_iter}{Integer, maximum iterations.}
\item{tol}{Stop when the step is at most \code{tol}.}
\item{accelerate}{If \code{TRUE} run FISTA, otherwise ISTA.}
\item{backtrack}{If \code{TRUE} use the backtracking line search.}
\item{eta}{Backtracking expansion factor.}
\item{g_fun}{Optional function \code{(x) -> float}, value of \code{g}.}
}
\description{
\code{accelerate=TRUE} is FISTA (Beck & Teboulle 2009 eq. 4.1-4.3), \code{FALSE} is plain ISTA (Sec. 2). \code{backtrack=TRUE} uses the paper\'s Sec. 4 backtracking line search on L.
}
\value{
A list with \code{estimate}, \code{x}, \code{fun},   \code{objective}, \code{iterations}, \code{L}, \code{accelerated},   \code{converged}, \code{method}.
}
\keyword{internal}
