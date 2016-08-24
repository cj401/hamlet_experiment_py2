library(mixtools)
library(MASS)
library(segmented)
library(reshape)

Ellipse_Plot <- function(obs, state, mean, cov, iteration_nums){
    plot(y~x, data=obs, col=state, pch=16, cex=.8, xlab=" ", ylab=" ",
    main=toString(iteration_nums))
    for (j in 1:nrow(obs)){
        contour <- ellipse(as.vector(mean[j,]), cov)
        lines(contour, col=state[j])
    }
}

relabel <- function(old_state){
    len <- length(old_state)
    unique_element <- unique(old_state)
    num_of_unique <- length(unique_element)
    new_state <- rep(0, len)
    for (k in 1:len){
        for (l in 1:num_of_unique){
            if(old_state[k] == unique_element[l]){
                new_state[k] = l
                break
            }
        }
    }
    return (new_state)
}


Visualization <- function(obs, z, h, mean_files, W, thetastar_files, output_path){
    dimensions <- ncol(obs)
    n_obs <- nrow(obs)
    iterations <- nrow(z)
    h <- as.matrix(h[,-1])
    W <- as.matrix(W)
    iteration_num <- as.vector(z[,1])
    z <- as.matrix(z[,-c(1:2)])
    pdf(output_path)
    if (dimensions == 2){
        obs <- rename(obs, c(V1="x", V2="y"))
        if (identical(mean_files, character(0))){
            for (i in 1:iterations){
                theta_star <- read.table(thetastar_files[i])
                thetastar <- as.matrix(cbind(theta_star,1))
                mean <- thetastar %*% W
                cov <- matrix(0, dimensions, dimensions)
                for (k in 1:dimensions){cov[k,k]=1./h[i,k]}
                state <- relabel(as.vector(z[i,]))
                Ellipse_Plot(obs, state, mean, cov, iteration_num[i])
            }
        }
        else{
            for (i in 1:iterations){
                mean <- as.matrix(read.table(mean_files[i]))
                cov <- matrix(0, dimensions, dimensions)
                for (k in 1:dimensions){cov[k,k]=1./h[i,k]}
                state <- relabel(as.vector(z[i,]))
                Ellipse_Plot(obs, state, mean, cov, iteration_num[i])
            }
        }
    }
    else{
        pca <- prcomp(obs, scales=TRUE)
        pca_scores <- as.data.frame(pca$x)
        transform <- cbind(as.vector(pca$rotation[,1]), as.vector(pca$rotation[,2]))
        pca_scores <- pca_scores[,1:2]
        pca_scores <- rename(pca_scores, c(PC1="x", PC2="y"))
        if (identical(mean_files, character(0))){
            for (i in 1:iterations){
                theta_star <- read.table(thetastar_files[i])
                thetastar <- as.matrix(cbind(theta_star,1))
                mean <- thetastar %*% W %*% transform
                ori_cov <- matrix(0, dimensions, dimensions)
                for (k in 1:dimensions){ori_cov[k,k]=1./h[i,k]}
                cov <- t(transform) %*% ori_cov %*% transform
                state <- relabel(as.vector(z[i,]))
                Ellipse_Plot(pca_scores, state, mean, cov, iteration_num[i])
            }
        }
        else{
            for (i in 1:iterations){
                mean <- as.matrix(read.table(mean_files[i])) %*% transform
                ori_cov <- matrix(0, dimensions, dimensions)
                for (k in 1:dimensions){ori_cov[k,k]=1./h[i,k]}
                cov <- t(transform) %*% ori_cov %*% transform
                state <- relabel(as.vector(z[i,]))
                Ellipse_Plot(pca_scores, state, mean, cov, iteration_num[i])
            }
        }
    }
    dev.off()
}

#import file
args <- commandArgs(trailingOnly = TRUE)
data_path <- toString(args[2])
results_path <- toString(args[1])
obs <- read.table(paste("data",data_path,"obs.txt",sep="/"))
z <- read.table(paste("results",results_path,"z.txt",sep="/"), skip=1)
h <- read.table(paste("results",results_path,"h.txt",sep="/"), skip=1)
mean_root <- paste("results",results_path,"mean_by_state",sep="/")
mean_files <- list.files(path=mean_root, pattern="*txt", full.names=TRUE)
W <- read.table(paste("results",results_path,"W/W.txt",sep="/"))
thetastar_root <- paste("results",results_path,"thetastar",sep="/")
thetastar_files <- list.files(path=thetastar_root, pattern="*txt", full.names=TRUE)
output_path <- paste("results",results_path,"Visualization.pdf",sep="/")

Visualization(obs, z, h, mean_files, W, thetastar_files, output_path)