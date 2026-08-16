% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_keystore_default_path}
\alias{.morie_keystore_default_path}
\title{CRAN policy: packages must not write outside tempdir() without}
\usage{
.morie_keystore_default_path()
}
\description{
explicit user opt-in. The keystore default path therefore resolves to a session-scoped tempdir() location; users who want persistent keys set MORIE_KEYSTORE_PATH (or pass path = ... explicitly to the keystore_create / load / store / wipe functions). morie 0.9.4 was archived from CRAN over this same issue; do not regress.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
\examples{
.morie_keystore_default_path()
}
