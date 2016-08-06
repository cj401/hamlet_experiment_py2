library(mixtools)
library(MASS)
library(segmented)

Visulization <- function(obs, Z, H, mean_files, output_path){
  dimensions <- ncol(obs)
  observations <- nrow(obs)
  iterations <- nrow(Z)
  h <- as.matrix(H[,2:dimensions+1])
  z <- as.matrix(Z[,3:iterations+2])
  iteration_num <- as.vector(Z[,1])
  #Two cases: two dimension and higher dimension
  if (dimensions = 2){
    print("Data is two dimensional.")
    pdf(output_path)
    for (i in 1:iterations){
      mean <- as.matrix(read.table(mean_files[i]))
      cov <- matrix(0, dimensions, dimensions)
      for (k in 1:dimensions){cov[k,k]=1./h[,k]}
      plot(V2~V1, data=obs, col=z[i,], pch=16, cex=.8, main=toString(iteration_num[i]))
      for (j in 1:observations){
        contour <- ellipse(as.vector(mean[j,]), cov)
        lines(contour, col=z[i,j])
      }
    }
  }
  else{
    print("Data is more than two dimensional, PCA is used!")
    pca <- prcomp(obs, scale.=TRUE)
    pca_scores <- as.data.frame(pca$x)
    transform <- cbind(as.vector(pca$rotation[,1]),as.vector(pca$rotation[,2]))
    pdf(output_path)
    for (i in 1:iterations){
      mean <- as.matrix(read.table(mean_files[i]))
      cov <- matrix(0, dimensions, dimensions)
      for (k in 1:dimensions){cov[k,k]=1./h[,k]}
      trans_cov <- t(transform) %*% cov %*% transform
      plot(PC2 ~ PC1, data=pca_scores, col=z[i,], pch=16, cex=.8, main=toString(iteration_num[i]))
      for (j in 1:observations){
        trans_mean <- as.vector(mean[j,] %*% transform)
        contour <- ellipse(as.vector(trans_mean), trans_cov)
        lines(contour, col=z[i,j])
      }
    }
  }
  dev.off()
}

#import file
experiment_name <- readline(prompt="Experiment name: ")
obs <- read.table(paste("data",experiment_name,"obs.txt",sep="/"))
Z <- read.table(paste("results",experiment_name,"z.txt",sep="/"), skip=1)
H <- read.table(paste("results",experiment_name,"h.txt",sep="/"), skip=1)
mean_root <- paste("results",experiment_name,"mean_by_state",sep="/")
mean_files <- list.files(path=mean_root, pattern="*txt", full.names=TRUE)
output_path <- paste("results",experiment_name,"Visualization.pdf",sep="/")

Visulization(obs, Z, H, mean_files, output_path)