library(mixtools)
library(MASS)
library(segmented)

#Function
PCA_Visulization <- function(obs,z,H,W,theta_list){
  #get basic info and transform W and H into matrix for matrix computation
  n_dimension <- ncol(obs)
  n_observation <- nrow(obs)
  n_iteration <- nrow(z)
  w <- as.matrix(W)
  h <- as.matrix(H[,2:13])
  #Two cases: higher dimension or two dimension
  if (n_dimension>2){
    print("Data is more than two dimensional, we use PCA.")
    pca_obs <- prcomp(obs, scale.=TRUE)
    scores_obs <- as.data.frame(pca_obs$x)
    transform <- cbind(as.vector(pca_obs$rotation[,1]),as.vector(pca_obs$rotation[,2]))
    pdf()
    for (i in 1:n_iteration){
      thetastar <- read.table(theta_list[i])
      theta_star <- as.matrix(cbind(thetastar,1))
      sigma <- matrix(0,n_dimension,n_dimension)
      for (k in 1:n_dimension){sigma[k,k]=sqrt(1./h[i,k])}
      new_sigma <- t(transform) %*% sigma %*% transform
      mean <- theta_star %*% w
      classification=t(z[i,3:402])
      plot(PC1~PC2,data=scores_obs,col=classification,pch=16,cex=.8)
      for (j in 1:n_observation){
        new_mean <- as.vector(mean[j,]) %*% transform
        contour <- ellipse(as.vector(new_mean),new_sigma)
        lines(contour,col=classification[j])
      }
    }
    dev.off()
  }
  else{
    print("Data is two dimensional.")
    pdf()
    for (i in 1:n_iteration){
      thetastar <- read.table(theta_list[i])
      theta_star <- as.matrix(cbind(thetastar,1))
      sigma <- matrix(0,n_dimension,n_dimension)
      for (k in 1:n_dimension){sigma[k,k]=sqrt(1./h[i,k])}
      mean <- theta_star %*% w
      classification=t(z[i,3:402])
      plot(V1~V2,data=obs,col=classification,pch=16,cex=.8)
      for (j in 1:n_observation){
        contour <- ellipse(as.vector(mean[j,]),sigma)
        lines(contour,col=classification[j])
      }
    }
    dev.off()
  }
}

#import file
obs_path <- readline(prompt="obs.txt's full path (*/obs.txt): ")
obs <- read.table(obs_path)
z_path <- readline(prompt="z.txt's full path (*/z.txt): ")
z <- read.table(z_path,skip=1)
H_path <- readline(prompt="h.txt's full path (*/h.txt): ")
H <- read.table(H_path,skip=1)
W_path <- readline(prompt="W.txt's full path (*/W.txt): ")
W <- read.table(W_path)
theta_path <- readline(prompt="Directory for thetastar file (*/thetastar/): ")
theta_list <- list.files(path=theta_path, pattern="*txt", full.names=TRUE)

#Call function
PCA_Visulization(obs,z,H,W,theta_list)
