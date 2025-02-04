#!/usr/bin/env Rscript
library(rdist)
library(movMF)
library(reticulate)
library(MLmetrics)

np <- import("numpy")
#x_s    <- np$load( "data/test16_x_.npy" )
#x_s    <- np$load( "data/test9_x_spca_.npy" )
#y_true <- np$load( "data/test_y_.npy" )

x_s    <- np$load( "data/X_s_4.npy" )
y_true <- np$load( "data/Y_s_4.npy" )
mu_s   <- np$load( "data/mu_s_4.npy" )

attr(x_s,"z") <- y_true
n_clusters <- length(unique(y_true))

model <- movMF(x_s, n_clusters, nruns = 1000, kappa = list(common = TRUE))
model

y_pred = predict(model)

y_table_1 <- table(True = y_true, Fitted = y_pred)
y_table_2 <- y_table_1[,max.col(y_table_1, 'first')]
colnames(y_table_2) <- seq_len(ncol(y_table_2))

y_table_1
y_table_2

sum(diag(y_table_2))/sum(y_table_2)

norm_vec <- function(x){ sqrt(sum(x^2)) }
kappa_s_pred <- rep(0, n_clusters)
mu_s_pred <- matrix(nrow = n_clusters, ncol = ncol(x_s))
for(n in 1:n_clusters){
   kappa_s_pred[n] <- norm_vec(model$theta[n,])
   mu_s_pred   [n,] <- model$theta[n,] / kappa_s_pred[n]
}
print('kappa_s')
kappa_s_pred

cosine_distance <- function(x ,y)
{
    return( 1 - sum(x*y)/sqrt(sum(x^2)*sum(y^2)) )
}

min.col <- function(m, ...) max.col(-m, ...)

cross_distance_s_1 <- cdist( mu_s, mu_s_pred, cosine_distance )
cross_distance_s_2 <- cross_distance_s_1[,min.col(cross_distance_s_1, 'first')]
min.col(cross_distance_s_1, 'first')

#colnames(table_2) <- seq_len(ncol(y_table_2))
cross_distance_s_2

diag( cross_distance_s_2 )
max(diag( cross_distance_s_2 ))
max(abs(kappa_s_pred - 64))
#y_true
#y_pred

#Accuracy(y_pred=y_pred, y_true=y_true)
