EXPERIMENT_ROOT = "../../../../"

matrify.data <- function(path, filename, skip = 0)
{
    library(dplyr)
    full.path <- paste(EXPERIMENT_ROOT, path, sep = "/")
    paste(full.path, filename, sep = "/") %>%
        read.table(skip = skip) %>%
            as.matrix()
}


plot.results <- function(out.file, data.dir, m1.dir, m2.dir = NULL, iteration = NULL, cluster.threshold = 0)
{
    library(RColorBrewer)
    library(stringr)
    y <- matrify.data(data.dir, "obs.txt")
    z.gt <- matrify.data(data.dir, "z.txt")
    X.gt <- matrify.data(data.dir, "mean.txt")
    brewer.pal(name = "Set3", n = 12) %>% palette()
    if(!is.null(m1.dir))
    {
        z.model1 <- matrify.data(m1.dir, "z.txt", skip = 1)[,-(1:2)]
        if(is.null(iteration)) iteration = nrow(z.model1)
        ndot.model1 <- matrify.data(m1.dir, "n_dot.txt", skip = 1)[,-1]
        X1.path <- paste(m1.dir, "X/", sep = "/")
        iteration.filename <-
            str_pad(as.character((iteration - 1) * 50), 5, side = "left", pad = "0") %>%
                paste(".txt", sep = "")
        X.model1 <- matrify.data(X1.path, iteration.filename)
        pdf(out.file, width = 8, height = 4)
    }
    if(!is.null(m2.dir))
    {
        z.model2 <- matrify.data(m2.dir, "z.txt", skip = 1)[,-(1:2)]
        X2.path <- paste(m2.dir, "X/", sep = "/")
        X.model2 <- matrify.data(X2.path, iteration.filename)
        ndot.model2 <- matrify.data(m2.dir, "n_dot.txt", skip = 1)[,-1]
        par(mfrow = c(2,2))
    }
    if(is.null(m1.dir) & is.null(m2.dir))
    {
        pdf(out.file, width = 6, height = 6)
    }
    plot(y, col = z.gt + 1)
    text(X.gt, labels = as.character(1:40), cex = 2, col = 1:40)
    if(!is.null(m1.dir))
    {
        plot(y, col = (z.model1[iteration,] %% 12) + 1)
        points(X.model1[ndot.model1[iteration,] > cluster.threshold,], pch = 16,
               col = which(ndot.model1[iteration,] > cluster.threshold) %% 12, cex = 2)
    }
    if(!is.null(m2.dir))
    {
        plot(y, col = (z.model2[iteration,] %% 12) + 1)
        points(X.model2[ndot.model2[iteration,] > cluster.threshold,], pch = 16,
               col = which(ndot.model2[iteration,] > cluster.threshold) %% 12, cex = 2)
    }
    dev.off()
}

