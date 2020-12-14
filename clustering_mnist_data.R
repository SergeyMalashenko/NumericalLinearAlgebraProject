library(movMF)
library(reticulate)

np <- import("numpy")
x  <- np$load( "data/X_s_16.npy" )
y1 <- np$load( "data/Y_s_16.npy" )
## Fit a von Mises-Fisher mixture with the "right" number of components,
## using 10 EM runs.
y2 <- movMF(x, 10, nruns = 100, kappa = "uniroot")
table(True = y1, Fitted = predict(y2))
y2

## Inspect the fitted parameters:y2
## Compare the fitted classes to the true ones:
#table(True = attr(x, "z"), Fitted = predict(y2))

