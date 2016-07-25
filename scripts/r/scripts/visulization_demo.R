library(mixtools)
library(MASS)
library(segmented)

#Function
PCA_Visualization <- function(
    data_path,
    results_path,
    obs.filename = "obs.txt",
    z.filename = "z.txt",
    h.filename = "h.txt",
    w.filename = "W/W.txt",
    thetastar.dirname = "thetastar"
)
{
    print("Reading in observations...")
    obs <- read.table(paste(data_path, obs.filename, sep = "/"))
    print("Reading state labels...")
    z <- read.table(paste(results_path, z.filename, sep = "/"), skip = 1)
    print("Reading noise variances...")
    H <- read.table(paste(results_path, h.filename, sep = "/"), skip = 1)
    print("Reading emission weights...")
    W <- read.table(paste(results_path, w.filename, sep = "/"))
    print("Reading latent state vectors...")
    theta_list <- list.files(path=paste(results_path, thetastar.dirname, sep = "/"),
                             pattern="*txt", full.names=TRUE)
    ## get basic info and transform W and H into matrix for matrix computation
    n_dimension <- ncol(obs)
    n_observation <- nrow(obs)
    n_iteration <- nrow(z)
    w <- as.matrix(W)
    h <- as.matrix(H[,2:ncol(H)])
    ## Two cases: higher dimension or two dimension
    if (n_dimension != 2)
    {
        print("Data is not two dimensional, we use PCA.")
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
            classification=t(z[i,3:ncol(z)])
            plot(PC1~PC2,data=scores_obs,col=classification,pch=16,cex=.8)
            for (j in 1:n_observation){
                new_mean <- as.vector(mean[j,]) %*% transform
                contour <- ellipse(as.vector(new_mean),new_sigma)
                lines(contour,col=classification[j])
            }
        }
        dev.off()
    } else{
        print("Data is two dimensional.")
        pdf()
        for (i in 1:n_iteration){
            thetastar <- read.table(theta_list[i])
            theta_star <- as.matrix(cbind(thetastar,1))
            sigma <- matrix(0,n_dimension,n_dimension)
            for (k in 1:n_dimension){sigma[k,k]=sqrt(1./h[i,k])}
            mean <- theta_star %*% w
            classification=t(z[i,3:ncol(z)])
            plot(V1~V2,data=obs,col=classification,pch=16,cex=.8)
            for (j in 1:n_observation){
                contour <- ellipse(as.vector(mean[j,]),sigma)
                lines(contour,col=classification[j])
            }
        }
        dev.off()
    }
}
