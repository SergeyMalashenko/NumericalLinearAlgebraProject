install.packages("HSAUR3", dependencies = TRUE)
install.packages("movMF" )

data("household", package = "HSAUR3")
x <- as.matrix(household[, c(1:2, 4)])
gender <- household$gender
theta <- rbind(female = movMF(x[gender == "female", ], k = 1)$theta,male = movMF(x[gender == "male", ], k = 1)$theta)
set.seed(2008)

vMFs <- lapply(1:5, function(K) movMF(x, k = K, control= list(nruns = 20)))
sapply(vMFs, BIC)
