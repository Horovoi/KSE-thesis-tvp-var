# Complementary model with hyperparemeters specification after Primiceri (2005)


# Load package
library(tidyverse)
library(BayesianTools)
library(bvarsv)
library(lubridate)
library(stargazer)
library(tseries)
library(feasts)
library(fable)
library(forecast)
library(devtools)
library(vars)
library(BVAR)
library(openxlsx)
library(janitor)
library(tsibble)
library(gridExtra)
library(plotly)
library(xtable)
library(rio)
library(dplyr)

# Import data
xlsx <- loadWorkbook('Data.xlsx')
sheet_names <- sheets(xlsx)


### Data preparation

## Create basic time series data frames
df_list <- list()

for(i in 1:length(sheet_names)) {
  df_list[[i]] <- readWorkbook(xlsx, sheet = i, detectDates = F, rows = c(1, 14:86))
}

for (i in 1:length(df_list)) {
  df_list[[i]][,1] <- convert_to_date(unlist(df_list[[i]][,1]), character_fun = lubridate::dmy)
  df_list[[i]] <- df_list[[i]] %>% mutate(Date = yearmonth(Date)) %>% as_tsibble(index = Date)
}

## Re-scale original data sets using different techniques
# Select mixed data (all variables in levels but CPI in the rate of change)
data_i <- 2

# using only interest rate
df_ts <- ts(df_list[[data_i]][,-1], start = c(2016,1), end = c(2022,1), freq = 12) # No scaling

df_scale <- scale(df_ts) # Normalized data
df_log <- cbind(log(df_ts[,c(1:2)]), df_ts[,3], log(df_ts[,c(4:5)])) # in natural logs but interest rate
df_log100 <- cbind(log(df_ts[,c(1:2)])*100, df_ts[,3], log(df_ts[,c(4:5)])*100) # in natural logs X 100 but interest rate
df_log_all <- cbind(log(df_ts[,c(1:5)]))  # in natural logs
df_log100_all <- cbind(log(df_ts[,c(1:5)])*100) # in natural logs X 100

# Deselect import and export price indexes
df_ts <- df_ts[,1:5]
df_scale <- df_scale[,1:5]

# Set column names
var_names <- c('ipi', 'cpi', 'i', 'neer', 'tot')
colnames(df_ts) <- var_names
colnames(df_scale) <- var_names
colnames(df_log) <- var_names
colnames(df_log100) <- var_names
colnames(df_log_all) <- var_names
colnames(df_log100_all) <- var_names

start_t <- 1
end_t <- 73

# Pack all variants for scaled data in one list
ts_list <- list(df_ts[start_t:end_t,], df_scale[start_t:end_t,], df_log[start_t:end_t,],
                df_log100[start_t:end_t,], df_log_all[start_t:end_t,], df_log100_all[start_t:end_t,])

n <- nrow(ts_list[[1]]) # Number of obs.
tau <- n # Number of obs. to use for training (equal to the whole sample)
scale_i <- 2 # Select normalized data for the further analysis
t <- ts(rbind(ts_list[[scale_i]][1:tau,], ts_list[[scale_i]])) # Add training set to the test set

### Model estimation

# Fix random seed
set.seed(7)

## Set parameters for the TVP-VAR model
# Set parameters for the Gibbs Sampler
lag_var <- 3 # Number of lags (selected based on the lowest DIC estimate)
thinfac <- 10 # Keep every n-th iteration
nrep <- 20000 # Number of draws used for estimation
nburn <- 20000 # Size of the burn-in sample (these draws are dropped)
itprint <- 5000 # Number of MCMC draws to display

# Set variance multipliers for:
k_B <- 4 # The distribution of betas
k_A <- 4 # The simultaneous relation matrix
k_sig <- 1 # The log standard errors

# Set prior beliefs about the amount of time variation in the estimates of:
k_Q <- 0.01 # The coefficients
k_S <- 0.1 # The covariances
k_W <- 0.01 # The volatilities

# Set a horizon for IRFs
nhor <- 36

# Select an optimal number of lags for the simple VAR model
VARselect(t[1:tau,], lag.max = 4) # p = 3

## Estimate simple OLS
fit0.ols <- VAR(df_scale, p = lag_var)
summary(fit0.ols)

## Simple VAR
# Responses of CPI to different shocks
var3_irf_list <- list()

for (i in 1:length(var_names)) {
  var3_irf_list[[i]] <- list()
  for (j in 1:length(var_names)) {
    var3_irf_list[[i]][[j]] <- vars::irf(fit0.ols,
                                         impulse = var_names[i],
                                         response = var_names[j],
                                         n.ahead = nhor,
                                         ci = 0.68)
  }
}

## Estimate the TVP-VAR model with 3 lags
fit1 <- bvar.sv.tvp(t, p = lag_var, tau = tau, pdrift = T, thinfac = thinfac,
                    nrep = nrep, nburn = nburn, itprint = itprint,
                    k_B = k_B, k_A = k_A, k_sig = k_sig, k_Q = k_Q, k_S = k_S, k_W = k_W)

# save the list with estimation results to reduce code running time later
saveRDS(fit1, "fit_tvp_rp.Rds")

# load model
#fit1 <- readRDS("fit_tvp_rp.Rds")

### Plot IRFs
## TVP-VAR

# Calculate IRFs with the 68% confidence interval
irf_list <- list()
irf_x_to_y_median <- list()
irf_x_to_y_16 <- list()
irf_x_to_y_84 <- list()

for (k in 1:ncol(t)) {
  irf_list[[k]] <- list()
  irf_x_to_y_median[[k]] <- list()
  irf_x_to_y_16[[k]] <- list()
  irf_x_to_y_84[[k]] <- list()
  for (j in 1:ncol(t)) {
    irf_list[[k]][[j]] <- list()
    irf_x_to_y_median[[k]][[j]] <- list()
    irf_x_to_y_16[[k]][[j]] <- list()
    irf_x_to_y_84[[k]][[j]] <- list()
    for (i in 1:(n-lag_var)) {
      irf_list[[k]][[j]][[i]] <- impulse.responses(fit1, impulse.variable = k,
                                                   response.variable = j, t = i, nhor = nhor,
                                                   scenario = 2, draw.plot = F)$irf
      irf_x_to_y_16[[k]][[j]][[i]] <- c(0, apply(as.matrix(irf_list[[k]][[j]][[i]]),
                                                 2, function(x) quantile(x, probs = c(.16))))
      irf_x_to_y_median[[k]][[j]][[i]] <- c(0, apply(as.matrix(irf_list[[k]][[j]][[i]]),
                                                     2, median))
      irf_x_to_y_84[[k]][[j]][[i]] <- c(0, apply(as.matrix(irf_list[[k]][[j]][[i]]),
                                                 2, function(x) quantile(x, probs = c(.84))))
    }
  }
}

# save the list with estimation results to reduce code running time later
#saveRDS(irf_list, "irf_list_rp.Rds")
#saveRDS(irf_x_to_y_median, "irf_x_to_y_median_rp.Rds")
#saveRDS(irf_x_to_y_16, "irf_x_to_y_16_rp.Rds")
#saveRDS(irf_x_to_y_84, "irf_x_to_y_84_rp.Rds")

# load model
irf_list <- readRDS("irf_list_rp.Rds")
irf_x_to_y_median <- readRDS("irf_x_to_y_median_rp.Rds")
irf_x_to_y_16 <- readRDS("irf_x_to_y_16_rp.Rds")
irf_x_to_y_84 <- readRDS("irf_x_to_y_84_rp.Rds")

## 1) Starting from different time points
# Create the vector of shock names
shock_names <- c("GDP", "CPI", "Interest Rate", "NEER", "Terms of Trade Index")
# Create the vector of names for time points
y_names <- as.character(df_list[[1]]$Date[(lag_var+1):n])

irf_plots <- list()

# For a note: COVID lockdowns
# 1st: 2020 Apr — 49
# 2nd: 2020 Aug — 53
# 3rd: 2021 Jan — 58
# 4th: 2021 Apr — 61

# Calculate mean IRFs
for (i in 1:length(irf_x_to_y_median)) {
  irf_plots[[i]] <- list()
  for(j in 1:length(irf_x_to_y_median)) {
    irf_plots[[i]][[j]] <- autoplot(ts(irf_x_to_y_median[[i]][[j]][[1]]), series = "2016 Apr") +
      autolayer(ts(irf_x_to_y_median[[i]][[j]][[49]]), series = "2020 Apr") +
      autolayer(ts(irf_x_to_y_median[[i]][[j]][[70]]), series = "2022 Jan") +
      autolayer(ts(rep(0, 36)), colour = F) +
      autolayer(ts(unlist(var3_irf_list[[i]][[j]]$irf)), series = "VAR(3)") +
      labs(title = paste0("Response of ", shock_names[j]),
           y = NULL) +
      scale_x_continuous(limits = c(0, 36), breaks = seq(0, 36, 6)) +
      theme_minimal()
  }
}

# Plot IRF starting from different time points (2016 Apr, 2020 Apr, 2022 Jan)
grid.arrange(grobs = c(irf_plots[[1]][2:5]), nrow = 2, ncol = 2) # GDP shocks
grid.arrange(grobs = c(irf_plots[[2]][c(1, 3:5)]), nrow = 2, ncol = 2) # CPI shocks
p_rp_1 <- grid.arrange(grobs = c(irf_plots[[3]][c(1:2, 4:5)]), nrow = 2, ncol = 2) # IR shocks
p_rp_2 <- grid.arrange(grobs = c(irf_plots[[4]][c(1:3, 5)]), nrow = 2, ncol = 2) # NEER shocks
grid.arrange(grobs = c(irf_plots[[5]][1:4]), nrow = 2, ncol = 2) # ToT shocks

# save plots
ggsave("irf_rp_ir.jpeg", plot = p_rp_1, units = "px", width = width, height = width/1.61803, device = 'jpeg', dpi = 300)
ggsave("irf_rp_neer.jpeg", plot = p_rp_2, units = "px", width = width, height = width/1.61803, device = 'jpeg', dpi = 300)

ggsave("irf_rp_gdp_cpi.jpeg", plot = irf_plots[[1]][[2]], units = "px", width = width, height = width/1.61803, device = 'jpeg', dpi = 300)


## 2) Responses to Interest rate shock with 68% confidence interval
irf_plots_ci <- list()

for (i in 1:length(irf_x_to_y_median[[3]])) {
  irf_plots_ci[[i]] <- list()
  for(j in 1:length(irf_x_to_y_median[[3]][[1]])) {
    irf_plots_ci[[i]][[j]] <- ggplot(data.frame("Time" = seq(1:37),
                                                "median" = irf_x_to_y_median[[3]][[i]][[j]],
                                                "p_16" = irf_x_to_y_16[[3]][[i]][[j]],
                                                "p_84" = irf_x_to_y_84[[3]][[i]][[j]]),
                                     aes(x = Time, y = median)) + 
      geom_line(col = 'red') + 
      geom_line(aes(y = ts(rep(0, 37)))) + 
      geom_ribbon(aes(ymin = p_16, ymax = p_84), alpha = 0.1) +
      labs(title = paste0("Response of ", shock_names[i]), y = NULL) +
      scale_x_continuous(limits = c(0, 36), breaks = seq(0, 36, 6)) +
      theme_minimal()
  }
}

# Plot responses to Interest rate shock at Jan 2022
time_point <- 1 # Jan 2022
grid.arrange(irf_plots_ci[[1]][[time_point]],
             irf_plots_ci[[2]][[time_point]],
             irf_plots_ci[[4]][[time_point]],
             irf_plots_ci[[5]][[time_point]],
             nrow = 2, ncol = 2)

## 3) Responses to Interest rate shock with 68% confidence interval at the n-th month  of shock
irf_fit_nth <- list()

nth <- c(1, 6, 12, 24, 36)

# Calculate
for (h in 1:length(nth)){
  irf_fit_nth[[h]] <- list()
  for (i in 1:length(irf_x_to_y_median[[3]])){
    irf_fit_nth[[h]][[i]] <- as.data.frame(matrix(nrow = n-lag_var, ncol = 3))
    for (k in 1:(n-lag_var)) {
      temp <- data.frame("median" = irf_x_to_y_median[[3]][[i]][[k]],
                         "p_16" = irf_x_to_y_16[[3]][[i]][[k]],
                         "p_84" = irf_x_to_y_84[[3]][[i]][[k]])
      irf_fit_nth[[h]][[i]][k,] <- temp[(1 + nth[h]),]
    }
    irf_fit_nth[[h]][[i]] <- cbind(y_names, irf_fit_nth[[h]][[i]])
    colnames(irf_fit_nth[[h]][[i]]) <- c('Date', 'median', 'p_16', 'p_84')
    irf_fit_nth[[h]][[i]] <- irf_fit_nth[[h]][[i]] %>% mutate(Date = yearmonth(Date)) %>%
      as_tsibble(index = Date)
  }
}

# Plot
irf_plots_nth <- list()

for (i in 1:length(irf_fit_nth)) {
  irf_plots_nth[[i]] <- list()
  for(j in 1:length(irf_fit_nth[[1]])) {
    irf_plots_nth[[i]][[j]] <- ggplot(irf_fit_nth[[i]][[j]], aes(x = Date, y = median)) +
      geom_line(col = 'red') + 
      geom_line(aes(y = ts(rep(0, 70)))) + 
      geom_ribbon(aes(ymin = p_16, ymax = p_84), alpha = 0.1) +
      labs(title = paste0("Response of ", shock_names[j], " to ", shock_names[3], " shock at ",
                          nth[i], "th month"), y = NULL) +
      theme_minimal()
  }
}

# Plot responses to Interest rate shock at the n-th month of shock
nth_point <- 3 # Responses at the 12-th month
grid.arrange(irf_plots_nth[[nth_point]][[1]],
             irf_plots_nth[[nth_point]][[2]],
             irf_plots_nth[[nth_point]][[4]],
             irf_plots_nth[[nth_point]][[5]],
             nrow = 2, ncol = 2)

## 4) IRFs for multiple time points plotted simultaneously (surface plots)
# Set axis
irf_fit_3d <- list()
m <- matrix(nrow = n-lag_var, ncol = nhor + 1)
axx <- list(title = "Response horizon")
axz <- list(title = "Response")
axy <- list(ticktext = y_names[c(1, seq(10, 70, 12))], tickvals = c(1, (seq(10, 70, 12))),
            range = c(0, 70), title = "Date of the shock")
camera <- list(eye = list(x = 1.75, y = -1.75, z = 0.55))

# Create the list of matrices with IRFs
for (i in 1:length(irf_x_to_y_median)) {
  irf_fit_3d[[i]] <- list()
  for(j in 1:length(irf_x_to_y_median)) {
    irf_fit_3d[[i]][[j]] <- m
    for (k in 1:nrow(m)) {
      irf_fit_3d[[i]][[j]][k,] <- unlist(irf_x_to_y_median[[i]][[j]][k])
    }
  }
}

# Create IRF surface plots
irf_plots_3d <- lapply(seq_len(length(irf_x_to_y_median)), function(j) {
  lapply(seq_len(length(irf_x_to_y_median)), function(i) {
    plot_ly(z = ~ irf_fit_3d[[j]][[i]]) %>% add_surface(showscale = FALSE) %>%
      layout(title = paste0(shock_names[i], " to ", shock_names[j], " shock"),
             scene = list(
               xaxis = axx,
               yaxis = axy,
               zaxis = axz,
               camera = camera,
               aspectmode = 'cube'))
  })
})

# Responses to Output shock
irf_plots_3d[[1]][[2]]
irf_plots_3d[[1]][[3]]
irf_plots_3d[[1]][[4]]
irf_plots_3d[[1]][[5]]

# Responses to CPI shock
irf_plots_3d[[2]][[1]]
irf_plots_3d[[2]][[3]]
irf_plots_3d[[2]][[4]]
irf_plots_3d[[2]][[5]]

# Responses to Interest rate shock
irf_plots_3d[[3]][[1]]
irf_plots_3d[[3]][[2]]
irf_plots_3d[[3]][[4]]
irf_plots_3d[[3]][[5]]

# Responses to NEER shock
irf_plots_3d[[4]][[1]]
irf_plots_3d[[4]][[2]]
irf_plots_3d[[4]][[3]]
irf_plots_3d[[4]][[5]]

# Responses to ToT shock
irf_plots_3d[[5]][[1]]
irf_plots_3d[[5]][[2]]
irf_plots_3d[[5]][[3]]
irf_plots_3d[[5]][[3]]


### Significance tests

## Calculate posterior probabilities for the difference in impulse responses to a IR shock

irf_mcmc_nth <- list()

# Build tables with MCMC draws at 2016 Apr, 2020 Apr and 2022 Jan
shock_prob <- 1 # calculate probabilities for GDP shocks
#shock_prob <- 3 # calculate probabilities for IR shocks
#shock_prob <- 4 # calculate probabilities for NEER shocks

for (h in 1:length(nth)){
  irf_mcmc_nth[[h]] <- list()
  for (i in 1:length(irf_x_to_y_median[[shock_prob]])){
    irf_mcmc_nth[[h]][[i]] <- data.frame(irf_list[[shock_prob]][[i]][[1]][,nth[h]],
                                         irf_list[[shock_prob]][[i]][[49]][,nth[h]],
                                         irf_list[[shock_prob]][[i]][[70]][,nth[h]])
    colnames(irf_mcmc_nth[[h]][[i]]) <- c(y_names[1], y_names[49],y_names[70])
  }
}

# Estimate probabilities
irf_prob_tbl <- list()

for (i in 1:length(irf_mcmc_nth[[1]])) {
  irf_prob_tbl[[i]] <- as.data.frame(matrix(nrow = 3, ncol = length(irf_mcmc_nth)))
  for (h in 1:length(irf_mcmc_nth)) {
    irf_prob_tbl[[i]][1,h] <- sum((irf_mcmc_nth[[h]][[i]][,1] < irf_mcmc_nth[[h]][[i]][,2]))/nrow(irf_mcmc_nth[[h]][[i]])
    irf_prob_tbl[[i]][2,h] <- sum((irf_mcmc_nth[[h]][[i]][,1] < irf_mcmc_nth[[h]][[i]][,3]))/nrow(irf_mcmc_nth[[h]][[i]])
    irf_prob_tbl[[i]][3,h] <- sum((irf_mcmc_nth[[h]][[i]][,2] < irf_mcmc_nth[[h]][[i]][,3]))/nrow(irf_mcmc_nth[[h]][[i]])
  }
  colnames(irf_prob_tbl[[i]]) <- c("M1", "M6", "M12", "M24", "M36")
  rownames(irf_prob_tbl[[i]]) <- c("2016/2020", "2016/2022", "2020/2022")
}

# Name each data frame inside lists with a respective name of response variable
names(irf_prob_tbl) <- var_names

## Wilcox test
irf_wilcox_tbl <- list()

for (i in 1:length(irf_mcmc_nth[[1]])) {
  irf_wilcox_tbl[[i]] <- as.data.frame(matrix(nrow = 3, ncol = length(irf_mcmc_nth)))
  for (h in 1:length(irf_mcmc_nth)) {
    irf_wilcox_tbl[[i]][1,h] <- wilcox.test(x = irf_mcmc_nth[[h]][[i]][,1], y = irf_mcmc_nth[[h]][[i]][,2])$p.value
    irf_wilcox_tbl[[i]][2,h] <- wilcox.test(x = irf_mcmc_nth[[h]][[i]][,1], y = irf_mcmc_nth[[h]][[i]][,3])$p.value
    irf_wilcox_tbl[[i]][3,h] <- wilcox.test(x = irf_mcmc_nth[[h]][[i]][,2], y = irf_mcmc_nth[[h]][[i]][,3])$p.value
  }
  colnames(irf_wilcox_tbl[[i]]) <- c("M1", "M6", "M12", "M24", "M36")
  rownames(irf_wilcox_tbl[[i]]) <- c("2016/2020", "2016/2022", "2020/2022")
}

# Name each data frame inside lists with a respective name of response variable
names(irf_wilcox_tbl) <- var_names

## Find significance levels for the Wilcox test
# Set indicators for significance levels using if-else construction
irf_wilcox_signif <- lapply(irf_wilcox_tbl, function(x){
  case_when(
    x < 0.001 ~ "***",
    x < 0.01 ~ "**",
    x < 0.05 ~ "*",
    x < 0.1 ~ ".",
    x < 1 ~ "n.s.")
})

# Build a table with significance levels
for (elem in 1:length(irf_wilcox_signif)) {
  irf_wilcox_signif[[elem]] <- as.data.frame(matrix(irf_wilcox_signif[[elem]], nrow = 3, ncol = length(irf_mcmc_nth)))
  colnames(irf_wilcox_signif[[elem]]) <- c("M1", "M6", "M12", "M24", "M36")
  rownames(irf_wilcox_signif[[elem]]) <- c("2016/2020", "2016/2022", "2020/2022")
}

# enumerate variables of interes
set_prob <- c(1,2,3,4,5)

# Show posterior probabilities for the difference in impulse responses
lapply(irf_prob_tbl[setdiff(set_prob, shock_prob)], function(x) {x*100})

# Plot a table with significance levels in impulse responses
irf_wilcox_signif[setdiff(set_prob, shock_prob)]



