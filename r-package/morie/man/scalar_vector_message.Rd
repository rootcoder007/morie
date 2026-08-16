% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{scalar_vector_message}
\alias{scalar_vector_message}
\title{The message: scalars from scalars, vectors from s*hat_r and s*v}
\usage{
scalar_vector_message(s_j, v_j, r_ij, phi_s, phi_v, W_rbf)
}
\arguments{
\item{s_j}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{v_j}{A matrix; passed to \code{as.matrix}.}
\item{r_ij}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{phi_s}{See Usage.}
\item{phi_v}{See Usage.}
\item{W_rbf}{See Usage.}
}
\description{
Type is preserved by construction, which is what makes the whole network equivariant rather than approximately so.
}
\value{
A list with \code{ds}, \code{dv}, \code{note}.
}
\keyword{internal}
