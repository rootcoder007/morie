% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gof_lillie_crit}
\alias{.gof_lillie_crit}
\title{Internal helper: Lilliefors critical values at sample size n}
\usage{
.gof_lillie_crit(n, dist)
}
\description{
Internal helper: Lilliefors critical values at sample size n
}
\value{
A vector.
}
\keyword{internal}
\examples{
set.seed(1)
mat <- matrix(0, nrow = 20, ncol = 3)
for (i in seq_len(20)) {
  pick <- sample.int(3, 5, replace = TRUE)
  for (k in pick) mat[i, k] <- mat[i, k] + 1
}
res <- fleiss_kappa(mat)
res$test_statistic
.gof_lillie_crit(n = i, dist = k)
}
