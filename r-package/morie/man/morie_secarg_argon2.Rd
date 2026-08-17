% Generated from the module source by ledger/gen_rd.R -- roxygen2 hangs
% on this package. Usage comes from args(), so it tracks the formals.
\name{morie_secarg_argon2}
\alias{morie_secarg_argon2}
\alias{morie_secarg}
\title{Argon2: make the attacker pay for memory, not just for time}
\usage{
morie_secarg_argon2(password, salt, memory = 32, passes = 3, parallelism = 4, tag_length = 32, variant = "argon2id", secret = NULL, associated = NULL)
}
\description{
Argon2: make the attacker pay for memory, not just for time. Sources: Biryukov, A., Dinu, D., Khovratovich, D. & Josefsson, S. design and trade-off analysis in Biryukov, Dinu & Khovratovich Native R arm mirroring the Python arm exactly: the same block permutation P applied to the 16-word rows and to the indexed columns, the same pre-hashing digest H_0 over every parameter so changing the variant, the memory, the passes or the lanes changes the tag, and the same data-independent addressing rule for Argon2i and for the first half of the first pass of Argon2id.
}
\value{
A list with components \code{argon2d}, \code{argon2i}, \code{argon2id}, \code{collapse}, \code{estimate}, \code{tag}, \code{tag_hex}, \code{variant}, \code{memory_kib}, \code{memory_used_kib}, \code{passes}, \code{parallelism}, \code{version}, \code{data_independent_first_half}, \code{y}, \code{method}, \code{note}.
}
\references{
(2021) "Argon2 Memory-Hard Function for Password Hashing and
Proof-of-Work Applications", RFC 9106, doi:10.17487/RFC9106; the
(2016) "Argon2", EuroS&P 292-302, doi:10.1109/EuroSP.2016.31; and
the BLAKE2 hash in Saarinen & Aumasson (2015) RFC 7693.
}
\keyword{internal}
\examples{
r <- morie_secarg_argon2(charToRaw("password"), charToRaw("somesalt"),
                         memory = 32, passes = 1)
str(r, max.level = 1)
}
