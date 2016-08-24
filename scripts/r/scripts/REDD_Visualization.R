Normal_Plot <- function(df, iteration_nums){
    y <- rep(0,nrow(df))
    plot(y~df$obs, col=df$relabel, ylim=c(0,1), cex=.8, ylab=" ", xlab=" ",
    main=toString(iteration_nums))
    for (i in 1:nrow(df)){
        curve(dnorm(x, mean=df$mean[i], sd=df$sd[i]), col=df$relabel[i], lwd=df$width[i],
        add=TRUE)
    }
}


Visulization <- function(obs, z, h, mean_files, n_dot, output_path){
    n_obs <- nrow(obs)
    iterations <- nrow(z)
    iteration_num <- as.vector(z[,1])
    obs <- as.vector(obs$V1)
    h <- as.vector(h$value)
    n_dot <- n_dot[,-1]
    last_column = 2 + n_obs
    z <- as.matrix(z[,-c(1:2)])
    pdf(output_path)
    for (i in 1:iterations){
        mean_by_state <- read.table(mean_files[i])
        mean <- as.vector(rowSums(mean_by_state))
        state <- as.vector(z[i,])
        sd <- rep(sqrt(1/h[i]),n_obs)
        df <- data.frame(obs,state,mean,sd)
        df$width <- NA
        for (j in 1:n_obs){
            col_label = df[j,]$state + 1
            df[j,]$width <- 1 + n_dot[i,col_label] * 0.025
        }
        df$relabel <- NA
        unique_element <- unique(df$state)
        num_of_unique <- length(unique_element)
        for (j in 1:n_obs){
            for (k in 1:num_of_unique){
                if(df[j,]$state == unique_element[k]){
                    df[j,]$relabel = k
                    break
                }
            }
        }
        Normal_Plot(df, iteration_num[i])
    }
    dev.off()
}


args <- commandArgs(trailingOnly = TRUE)
data_path <- toString(args[2])
results_path <- toString(args[1])
obs <- read.table(paste("data",data_path,"obs.txt",sep="/"))
z <- read.table(paste("results",results_path,"z.txt",sep="/"), skip=1)
h <- read.table(paste("results",results_path,"h.txt",sep="/"), header=TRUE)
n_dot <- read.table(paste("results",results_path,"n_dot.txt",sep="/"),skip=1)
mean_root <- paste("results",results_path,"mean_by_state",sep="/")
mean_files <- list.files(path=mean_root, pattern="*txt", full.names=TRUE)
output_path <- paste("results",results_path,"Visualization.pdf",sep="/")

Visulization(obs, z, h, mean_files, n_dot, output_path)