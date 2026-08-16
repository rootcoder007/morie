% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.val_score}
\alias{.val_score}
\title{Internal helper: Val Score}
\usage{
.val_score(scoring, y_true, y_pred)
}
\description{
Internal helper: Val Score
}
\value{
The value of \code{switch}.
}
\keyword{internal}
\examples{
child <- data.frame(fk = c(1, 2, 3))
parent <- data.frame(pk = c(1, 2, 3, 4))
res <- check_referential_integrity(child, parent, "fk", "pk")
res$passed
.val_score(scoring = child, y_true = child, y_pred = child)
}
