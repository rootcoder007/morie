% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_bq_billing_project}
\alias{.morie_bq_billing_project}
\title{Internal: resolve the billing project}
\usage{
.morie_bq_billing_project(billing_project = NULL)
}
\arguments{
\item{billing_project}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
}
\description{
A step of the ingest_bigquery implementation. Called by \code{morie_ingest_bigquery_query}. See the file header for the source the module follows.
}
\value{
Nothing; the function is called for its effect.
}
\keyword{internal}
\examples{
.morie_bq_billing_project()
}
