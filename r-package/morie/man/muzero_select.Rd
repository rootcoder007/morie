% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{muzero_select}
\alias{muzero_select}
\title{muzero_select}
\usage{
muzero_select(node, A_keys, mm, c1, c2)
}
\arguments{
\item{node}{A list; the body reads \code{$children} from it.}
\item{A_keys}{A vector; indexed elementwise.}
\item{mm}{A list; the body reads \code{$normalize} from it.}
\item{c1}{Numeric; combined arithmetically in the body.}
\item{c2}{Numeric; combined arithmetically in the body.}
}
\description{
Part of the muzero_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{best_a}, as built in the body.
}
\keyword{internal}
\examples{
root <- muzero_Node()
root$expand("s0", c(0.4, 0.3, 0.3), list(0, 1, 2))
mm <- muzero_MinMax()
a <- muzero_select(root, c("0", "1", "2"), mm, c1 = 1.25, c2 = 19652)
a \%in\% c("0", "1", "2")
}
