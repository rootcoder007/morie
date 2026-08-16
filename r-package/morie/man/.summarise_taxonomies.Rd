% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.summarise_taxonomies}
\alias{.summarise_taxonomies}
\title{Internal: compute counts + flag lists from a flat taxonomy list}
\usage{
.summarise_taxonomies(taxonomies, analyzed_set, domain)
}
\arguments{
\item{taxonomies}{A vector; its length is taken.}
\item{analyzed_set}{See Usage.}
\item{domain}{Character; passed to \code{toupper}.}
}
\description{
A step of the audit_variables implementation. Called by \code{morie_audit_arsau_variables}, \code{morie_audit_otis_variables}. See the file header for the source the module follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
