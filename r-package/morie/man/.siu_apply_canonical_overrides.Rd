% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_apply_canonical_overrides}
\alias{.siu_apply_canonical_overrides}
\title{Internal: apply a canonical-overrides table to a parsed SIU data}
\usage{
.siu_apply_canonical_overrides(df, overrides)
}
\arguments{
\item{df}{See Usage.}
\item{overrides}{See Usage.}
}
\description{
frame. Each row of `overrides` is (case_number, field, verified_value); for any match, overwrite df[[field]] at the row whose case_number matches. Silent on misses (override for a case not in the parse, or field not in the schema).
}
\value{
The value of \code{df}, as built in the body.
}
\keyword{internal}
