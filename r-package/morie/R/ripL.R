# R arm of morie/fn/ripL.py -- Besag's L function, centred.
#
# The Python body was a placeholder: it averaged the leading `points`
# argument and used neither `window` nor `r`. There was no R arm at all.
#
#   L(d) = sqrt(K(d) / pi) - d
#
# Under CSR K(d) = pi d^2, so this curve is identically zero; positive at
# short distances means clustering, negative means regularity.
#
# Besag, J. (1977). Discussion of "Modelling spatial patterns" by B. D.
# Ripley. JRSS B 39(2):193-195 (Ripley's paper is 39(2):172-192,
# doi:10.1111/j.2517-6161.1977.tb01615.x). Baddeley & Turner (2005) JSS
# 12(6) p. 17 states L(r) = sqrt(K-hat(r)/pi).
#
# `l` here is the CENTRED curve; splfun() returns the uncentred sqrt(K/pi)
# as `l` and the centred one as `l_minus_r`. They agree term by term.

#' @noRd
morie_ripley_l_function <- function(points, window = NULL, r = NULL,
                                    correction = "border") {
  reg <- .sp_region(window, as.matrix(points))
  res <- splfun(points, NULL, r, reg, correction)
  list(r = res$r, l = res$l_minus_r, l_uncentred = res$l, k = res$k,
       lambda_est = res$lambda_est,
       method = "Besag L function, sqrt(K(d)/pi) - d")
}

#' @noRd
RipL <- morie_ripley_l_function
