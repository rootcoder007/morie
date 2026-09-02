% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_tmle_dynamic_regime}
\alias{morie_tmle_dynamic_regime}
\title{Mean outcome under the (V-)optimal dynamic treatment rule}
\usage{
morie_tmle_dynamic_regime(y, treatment_history, covariate_history, regime = "optimal", 
    method = "cv-tmle", n_folds = 10, V0 = NULL, V1 = NULL, trim = 0.01, 
    known_g = NULL, ridge = 1e-08, level = 0.95) 
}
\arguments{
\item{y}{Numeric outcome vector.}
\item{treatment_history}{n-by-2 matrix of \code{A(0), A(1)}.}
\item{covariate_history}{Two-block list \code{list(L0, L1)} of   length n each.}
\item{regime}{Either \code{"optimal"} or a supplied rule.}
\item{method}{One of \code{"cv-tmle"}, \code{"tmle"}, \code{"ipw"},   \code{"gcomp"}.}
\item{n_folds}{Number of CV folds.}
\item{V0,V1}{Optional summaries the rules may depend on.}
\item{trim}{Positivity trim.}
\item{known_g}{Optional \code{list(p0, p1)} of known treatment   probabilities.}
\item{ridge}{Optional ridge.}
\item{level}{Confidence level.}
}
\description{
Mean outcome under the (V-)optimal dynamic treatment rule
}
\value{
A list with the mean outcome, SE, CI, the rule, blips, and   the four static comparators.
}
\keyword{internal}
\examples{
set.seed(2)
n <- 80
L0 <- rnorm(n)
A0 <- rbinom(n, 1, 0.5)
L1 <- rnorm(n) + 0.4 * A0
A1 <- rbinom(n, 1, 0.5)
y <- plogis(0.3 * A0 + 0.5 * A1 + 0.4 * L0 + rnorm(n, 0, 0.4))
r <- morie_tmle_dynamic_regime(y, cbind(A0, A1), list(L0, L1),
                               method = "tmle")
str(r, max.level = 1)
}
