% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{nashq_run}
\alias{nashq_run}
\title{nashq_run}
\usage{
nashq_run(states, actions, step, rewards, gamma = 0.9, alpha = 0.5, epsilon = 0.1, 
    episodes = 500, horizon = 50, start = NULL, selection = "global_optimal", 
    terminal = NULL, seed = 0, agent = 0, tol = 1e-09) 
}
\arguments{
\item{states}{Coerced to list by the body, with \code{as.list}.}
\item{actions}{A vector; its length is taken and its elements indexed.}
\item{step}{A function; the body checks with \code{is.function}.}
\item{rewards}{A function; the body checks with \code{is.function}.}
\item{gamma}{Numeric; combined arithmetically in the body. Defaults to \code{0.9}.}
\item{alpha}{Numeric; combined arithmetically in the body. Defaults to \code{0.5}.}
\item{epsilon}{Passed to \code{nashq_pick}. Defaults to \code{0.1}.}
\item{episodes}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{500}.}
\item{horizon}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{50}.}
\item{start}{Optional; may be \code{NULL}. A function; the body checks with \code{is.function}.}
\item{selection}{Carried through into a list the body builds. Defaults to \code{"global_optimal"}.}
\item{terminal}{Optional; may be \code{NULL}. Coerced to list by the body, with \code{as.list}.}
\item{seed}{Passed to \code{set.seed}. Defaults to \code{0}.}
\item{agent}{Passed to \code{nashq_select_eq}. Defaults to \code{0}.}
\item{tol}{Passed to \code{nashq_select_eq}. Defaults to \code{1e-09}.}
}
\description{
Part of the nashq_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{q}, \code{policy}, \code{nash_values}, \code{stage_game_types}, \code{returns}, \code{mean_return_last}, \code{selection}, \code{method}.
}
\keyword{internal}
\examples{
A <- matrix(c(2, 0, 0, 1), 2, 2, byrow = TRUE)
B <- matrix(c(1, 0, 0, 2), 2, 2, byrow = TRUE)
res <- nashq_run(list("s"), list(0:1, 0:1),
                 function(s, a1, a2) "s",
                 function(s, a1, a2, s1) c(A[a1 + 1, a2 + 1],
                                           B[a1 + 1, a2 + 1]),
                 episodes = 20, horizon = 5, seed = 1)
res$selection
}
