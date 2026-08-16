% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gsage_rows}
\alias{.gsage_rows}
\title{Permutation-invariant neighbour aggregation}
\usage{
.gsage_rows(x)
}
\arguments{
\item{vectors}{A list of numeric vectors.}
\item{how}{One of \code{"mean"}, \code{"max_pool"}, \code{"lstm_order"}.}
\item{W}{Optional weight matrix used by \code{max_pool}.}
}
\description{
The three aggregators of Hamilton et al. 2017: \code{mean} (the default, nearly the transductive GCN\'s rule), \code{max_pool} (pass each neighbour through a linear layer and take an element-wise max), and \code{lstm_order} (returns the vectors in the order given -- the paper\'s point that an LSTM aggregator is not symmetric and must be fed a random permutation).
}
\value{
A single numeric vector.
}
\keyword{internal}
