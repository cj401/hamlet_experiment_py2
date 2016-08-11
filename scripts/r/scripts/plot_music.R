RESULTS_ROOT = "~/projects/hamlet/experiment/results"

noLT.path <- paste(RESULTS_ROOT, "/tmp/music_noLT/", sep = "")
LT.path <- paste(RESULTS_ROOT, "/tmp/music_LT/", sep = "")

X.file <- "X/05000.txt"

raw.data <- as.matrix(read.table("~/projects/hamlet/experiment/data/music/obs.txt"))
library(mosaic)
p <- tally(raw.data, format = "proportion")

X.noLT <- as.matrix(read.table(paste(noLT.path, X.file, sep = "")))
X.LT <- as.matrix(read.table(paste(LT.path, X.file, sep = "")))

X.noLT <- sweep(X.noLT, STATS = p, FUN = `/`, MARGIN = 2)
X.LT <- sweep(X.LT, STATS = p, FUN = `/`, MARGIN = 2)

n.noLT <- as.matrix(read.table(paste(noLT.path, "n_dot.txt", sep = ""), skip = 1))[,-1]
n.LT <- as.matrix(read.table(paste(LT.path, "n_dot.txt", sep = ""), skip = 1))[,-1]
j.noLT <- which(n.noLT[nrow(n.noLT),] > 0)
j.LT <- which(n.noLT[nrow(n.noLT),] > 0)

LL.noLT <- as.matrix(read.table(paste(noLT.path, "train_log_likelihood.txt", sep = ""), skip = 1))[,-1]
LL.LT <- as.matrix(read.table(paste(LT.path, "train_log_likelihood.txt", sep = ""), skip = 1))[,-1]

X.noLT.thresh <- X.noLT[j.noLT,] > 3
X.LT.thresh <- X.LT[j.noLT,] > 3

par(mfrow = c(2,1), mar = c(4,4,1,1))
image(t(X.noLT.thresh), xaxt = "n", yaxt = "n", xlab = "Chords", ylab = "Latent States",
      col = heat.colors(100))
axis(side = 2, at = seq(0,1, length = length(j.noLT)), labels = 1:length(j.noLT))
image(t(X.LT.thresh), xaxt = "n", yaxt = "n", xlab = "Chords", ylab = "Latent States",
      col = heat.colors(100))
axis(side = 2, at = seq(0,1, length = length(j.LT)), labels = 1:length(j.LT))




