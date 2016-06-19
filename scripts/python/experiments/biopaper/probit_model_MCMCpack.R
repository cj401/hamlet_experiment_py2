library(MASS)  # birthwt data set
library(MCMCpack)

data(birthwt)
posterior <- MCMClogit(low ~ age + as.factor(race) + smoke, data=birthwt)

out1 <- MCMCprobit(low~as.factor(race)+smoke, data=birthwt,
                   b0 = 0, B0 = 10, marginal.likelihood="Chib95")
out2 <- MCMCprobit(low~age+as.factor(race), data=birthwt,
                   b0 = 0, B0 = 10, marginal.likelihood="Chib95")
out3 <- MCMCprobit(low~age+as.factor(race)+smoke, data=birthwt,
                   b0 = 0, B0 = 10, marginal.likelihood="Chib95")
BayesFactor(out1, out2, out3)
plot(out3)
summary(out3)
