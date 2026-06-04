install.packages(c("xgboost","Matrix","pROC","caret","dplyr"))
library(xgboost)
library(Matrix)
library(pROC)
library(caret)
library(dplyr)
library(ggplot2)
library(PRROC)
set.seed(123)

# ---------------------------
# 0) Minimal cleaning helpers
# ---------------------------

to01 <- function(x) {
  if (is.factor(x)) x <- as.character(x)
  if (is.logical(x)) return(as.numeric(x))
  if (is.character(x)) {
    x <- trimws(x)
    x <- ifelse(toupper(x) %in% c("TRUE","T","YES","Y","1"), 1,
                ifelse(toupper(x) %in% c("FALSE","F","NO","N","0"), 0, NA))
    return(as.numeric(x))
  }
  return(as.numeric(x))
}

cat_missing <- function(x) {
  if (!is.factor(x)) x <- as.factor(x)
  x <- as.character(x)
  x[is.na(x) | x == ""] <- "Missing"
  factor(x)
}

impute_num_with_flag <- function(df, varname) {
  v <- df[[varname]]
  miss_flag <- is.na(v)
  med <- median(v, na.rm = TRUE)
  v[miss_flag] <- med
  df[[varname]] <- v
  df[[paste0(varname, "_miss")]] <- as.integer(miss_flag)
  df
}

# -----------------------------------
# 1) Define features 
# -----------------------------------
feature_vars <- c(
  "gender","race","brain_death","opo",
  "age","referral_year",
  "tissue_referral","eye_referral",
  "height_in","weight_kg",
  "mechanism_of_death","circumstances_of_death"
)

# ---------------------------------------------------------
# 2) Load data + type conversions
# ---------------------------------------------------------
dat <- read.csv(file.choose())
str(dat)

# Coerce to factor for categoricals first (safe even if already factor)
dat$gender <- as.factor(dat$gender)
dat$opo <- as.factor(dat$opo)
dat$race <- as.factor(dat$race)
dat$brain_death <- as.factor(dat$brain_death)

dat$tissue_referral <- as.factor(dat$tissue_referral)
dat$eye_referral <- as.factor(dat$eye_referral)

dat$mechanism_of_death <- as.factor(dat$mechanism_of_death)
dat$circumstances_of_death <- as.factor(dat$circumstances_of_death)

# Outcomes to numeric 0/1
dat$approached  <- to01(dat$approached)
dat$authorized  <- to01(dat$authorized)
dat$procured    <- to01(dat$procured)

# Categorical: add "Missing"
for (v in c("gender","race","brain_death","opo","mechanism_of_death","circumstances_of_death")) {
  if (v %in% names(dat)) dat[[v]] <- cat_missing(dat[[v]])
}

# tissue_referral, eye_referral: make "0","1","Missing"
for (v in c("tissue_referral","eye_referral")) {
  if (v %in% names(dat)) {
    x <- dat[[v]]
    if (is.factor(x)) x <- as.character(x)
    x <- as.character(x)
    x[is.na(x) | x == ""] <- "Missing"
    # standardize typical encodings
    x <- ifelse(x %in% c("0","1","Missing"), x, x)
    dat[[v]] <- factor(x, levels = c("0","1","Missing"))
  }
}

# Numeric imputations + missing flags
for (v in c("age","height_in","weight_kg","referral_year")) {
  if (v %in% names(dat)) dat <- impute_num_with_flag(dat, v)
}

# Updated feature list includes missing flags if present
feature_vars2 <- unique(c(
  feature_vars,
  "age_miss","height_in_miss","weight_kg_miss","referral_year_miss"
))
feature_vars2 <- feature_vars2[feature_vars2 %in% names(dat)]

# ---------------------------
# 3) Train/test split ONCE (do not filter outcomes here)
# ---------------------------
# Use a stable key for splitting: patient_id if available; else row index
# Also ensure procured is not NA for stratification; if many NAs, fallback to simple split
if ("procured" %in% names(dat) && sum(!is.na(dat$procured)) > 1000) {
  dat_split <- dat %>% filter(!is.na(procured))
  idx <- createDataPartition(dat_split$procured, p = 0.8, list = FALSE)
  train_df_base <- dat_split[idx, ]
  test_df_base  <- dat_split[-idx, ]
} else {
  idx <- createDataPartition(rep(1, nrow(dat)), p = 0.8, list = FALSE)
  train_df_base <- dat[idx, ]
  test_df_base  <- dat[-idx, ]
}

# -------------------------------------------------------
# 4) Function: fit XGBoost and compute key metrics
# -------------------------------------------------------
fit_xgb_binary <- function(train_df, test_df, outcome_var, feature_vars) {
  
  # ---- IMPORTANT FIX: use identical rows for X and y ----
  keep_tr <- complete.cases(train_df[, c(feature_vars, outcome_var)])
  train_use <- train_df[keep_tr, ]
  
  keep_te <- complete.cases(test_df[, c(feature_vars, outcome_var)])
  test_use <- test_df[keep_te, ]
  
  # y must be numeric 0/1
  y_train <- as.numeric(train_use[[outcome_var]])
  y_test  <- as.numeric(test_use[[outcome_var]])
  
  # Build sparse one-hot matrix from the same filtered data
  fml <- as.formula(paste0("~ ", paste(feature_vars, collapse = " + "), " - 1"))
  
  X_train <- sparse.model.matrix(fml, data = train_use)
  X_test  <- sparse.model.matrix(fml, data = test_use)
  
  # sanity checks (this is the exact error you got)
  stopifnot(nrow(X_train) == length(y_train))
  stopifnot(nrow(X_test)  == length(y_test))
  
  dtrain <- xgb.DMatrix(data = X_train, label = y_train)
  dtest  <- xgb.DMatrix(data = X_test,  label = y_test)
  
  # Class imbalance handling
  n_pos <- sum(y_train == 1, na.rm = TRUE)
  n_neg <- sum(y_train == 0, na.rm = TRUE)
  spw <- ifelse(n_pos > 0, n_neg / n_pos, 1)
  
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.05,
    max_depth = 5,
    min_child_weight = 5,
    subsample = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = spw
  )
  
  watch <- list(train = dtrain, test = dtest)
  
  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 2000,
    watchlist = watch,
    early_stopping_rounds = 50,
    verbose = 0
  )
  
  # Probabilities on test set
  prob <- predict(model, dtest)
  
  # AUC
  roc_obj <- roc(y_test, prob, quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  
  # Best threshold (Youden J)
  coords_best <- coords(
    roc_obj, x = "best", best.method = "youden",
    ret = c("threshold","sensitivity","specificity"),
    transpose = FALSE
  )
  best_thresh <- as.numeric(coords_best["threshold"])
  
  # Pred class
  pred <- ifelse(prob >= best_thresh, 1, 0)
  
  cm <- confusionMatrix(
    factor(pred, levels = c(0,1)),
    factor(y_test, levels = c(0,1)),
    positive = "1"
  )
  
  list(
    model = model,
    outcome = outcome_var,
    auc = auc_val,
    best_thresh = best_thresh,
    confusion = cm,
    prob_test = prob,
    y_test = y_test,
    pred_test = pred,
    test_use = test_use  # return filtered test rows for subgroup fairness
  )
}

# -------------------------------------------------------
# 5) Fairness metrics function (on TEST set only)
# -------------------------------------------------------
fairness_metrics_test <- function(test_df_used, y_true, y_prob, y_pred, subgroup_var) {
  
  groups <- sort(unique(test_df_used[[subgroup_var]]))
  out <- list()
  
  for (g in groups) {
    if (is.na(g) || as.character(g) == "") next
    idx <- which(test_df_used[[subgroup_var]] == g)
    if (length(idx) == 0) next
    
    y_t <- y_true[idx]
    y_p <- y_pred[idx]
    y_pr <- y_prob[idx]
    
    auc_val <- NA
    if (length(unique(y_t)) == 2) {
      roc_obj <- roc(y_t, y_pr, quiet = TRUE)
      auc_val <- as.numeric(auc(roc_obj))
    }
    
    cm <- confusionMatrix(
      factor(y_p, levels = c(0,1)),
      factor(y_t, levels = c(0,1)),
      positive = "1"
    )
    
    ppr <- mean(y_p == 1)
    
    out[[as.character(g)]] <- data.frame(
      Group = as.character(g),
      N = length(y_t),
      Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
      Specificity = as.numeric(cm$byClass["Specificity"]),
      Precision = as.numeric(cm$byClass["Pos Pred Value"]),
      PPR = ppr,
      AUC = auc_val,
      row.names = NULL
    )
  }
  
  do.call(rbind, out)
}

fairness_gaps <- function(results_df) {
  metrics <- c("Sensitivity","Specificity","Precision","PPR","AUC")
  gaps <- sapply(metrics, function(m) {
    max(results_df[[m]], na.rm = TRUE) - min(results_df[[m]], na.rm = TRUE)
  })
  data.frame(Metric = metrics, Gap = round(gaps, 3))
}

# -------------------------------------------------------
# 6) Fit models for all 3 outcomes
#    IMPORTANT: Each outcome uses its own non-NA subset internally
# -------------------------------------------------------
res_approached <- fit_xgb_binary(train_df_base, test_df_base, "approached", feature_vars2)
res_authorized <- fit_xgb_binary(train_df_base, test_df_base, "authorized", feature_vars2)
res_procured   <- fit_xgb_binary(train_df_base, test_df_base, "procured", feature_vars2)

# Print summaries
res_approached$auc
res_authorized$auc
res_procured$auc

res_approached$best_thresh
res_authorized$best_thresh
res_procured$best_thresh

res_approached$confusion
res_authorized$confusion
res_procured$confusion

# -------------------------------------------------------
# 7) Fairness evaluation (TEST set, outcome-specific filtered rows)
# -------------------------------------------------------
# Add age_bin to each filtered test set
res_approached$test_use$age_bin <- cut(
  res_approached$test_use$age,
  c(0,39,64,200),
  labels = c("<40","40-64","65+"),
  include.lowest = TRUE
)

res_authorized$test_use$age_bin <- cut(
  res_authorized$test_use$age,
  c(0,39,64,200),
  labels = c("<40","40-64","65+"),
  include.lowest = TRUE
)

res_procured$test_use$age_bin <- cut(
  res_procured$test_use$age,
  c(0,39,64,200),
  labels = c("<40","40-64","65+"),
  include.lowest = TRUE
)

# APPROACHED fairness
fair_race_app   <- fairness_metrics_test(res_approached$test_use, res_approached$y_test, res_approached$prob_test, res_approached$pred_test, "race")
fair_gender_app <- fairness_metrics_test(res_approached$test_use, res_approached$y_test, res_approached$prob_test, res_approached$pred_test, "gender")
fair_age_app    <- fairness_metrics_test(res_approached$test_use, res_approached$y_test, res_approached$prob_test, res_approached$pred_test, "age_bin")

fairness_gaps(fair_race_app)
fairness_gaps(fair_gender_app)
fairness_gaps(fair_age_app)

# AUTHORIZED fairness
fair_race_authz   <- fairness_metrics_test(res_authorized$test_use, res_authorized$y_test, res_authorized$prob_test, res_authorized$pred_test, "race")
fair_gender_authz <- fairness_metrics_test(res_authorized$test_use, res_authorized$y_test, res_authorized$prob_test, res_authorized$pred_test, "gender")
fair_age_authz    <- fairness_metrics_test(res_authorized$test_use, res_authorized$y_test, res_authorized$prob_test, res_authorized$pred_test, "age_bin")

fairness_gaps(fair_race_authz)
fairness_gaps(fair_gender_authz)
fairness_gaps(fair_age_authz)

# PROCURED fairness
fair_race_proc   <- fairness_metrics_test(res_procured$test_use, res_procured$y_test, res_procured$prob_test, res_procured$pred_test, "race")
fair_gender_proc <- fairness_metrics_test(res_procured$test_use, res_procured$y_test, res_procured$prob_test, res_procured$pred_test, "gender")
fair_age_proc    <- fairness_metrics_test(res_procured$test_use, res_procured$y_test, res_procured$prob_test, res_procured$pred_test, "age_bin")

fairness_gaps(fair_race_proc)
fairness_gaps(fair_gender_proc)
fairness_gaps(fair_age_proc)

# -------------------------------------------------------
# 8) Feature importance 
# -------------------------------------------------------
imp_proc <- xgb.importance(model = res_procured$model)
head(imp_proc, 20)
xgb.plot.importance(imp_proc[1:20,], main = "Top 20 Feature Importances (Procured)")

imp_appr <- xgb.importance(model = res_approached$model)
head(imp_appr, 20)
xgb.plot.importance(imp_appr[1:20,], main = "Top 20 Feature Importances (Approched)")

imp_auth <- xgb.importance(model = res_authorized$model)
head(imp_auth, 20)
xgb.plot.importance(imp_auth[1:20,], main = "Top 20 Feature Importances (Authorized)")

#####################################################################################33333
   
#1) PR-AUC (AUPRC) functions
# ----------------------------
pr_auc <- function(y_true, y_prob) {
  # y_true must be 0/1 numeric
  y_true <- as.numeric(y_true)
  ok <- !is.na(y_true) & !is.na(y_prob)
  y_true <- y_true[ok]
  y_prob <- y_prob[ok]
  if (length(unique(y_true)) < 2) return(NA_real_)
  fg <- y_prob[y_true == 1]
  bg <- y_prob[y_true == 0]
  if (length(fg) == 0 || length(bg) == 0) return(NA_real_)
  pr <- PRROC::pr.curve(scores.class0 = fg, scores.class1 = bg, curve = FALSE)
  as.numeric(pr$auc.integral)
}
res_approached$auprc  <- pr_auc(res_approached$y_test,  res_approached$prob_test)
res_authorized$auprc  <- pr_auc(res_authorized$y_test,  res_authorized$prob_test)
res_procured$auprc    <- pr_auc(res_procured$y_test,    res_procured$prob_test)

res_approached$auprc
res_authorized$auprc
res_procured$auprc

#2) Calibration check overall and by subgroup
# ---------------------------------------------
# Metrics: Brier score + simple calibration table/plot
brier_score <- function(y_true, y_prob) {
  y_true <- as.numeric(y_true)
  ok <- !is.na(y_true) & !is.na(y_prob)
  y_true <- y_true[ok]
  y_prob <- y_prob[ok]
  mean((y_prob - y_true)^2)
}

# Create calibration bins and summarize observed vs predicted
calibration_table <- function(df, y_true, y_prob, n_bins = 10) {
  tmp <- df
  tmp$.y <- as.numeric(y_true)
  tmp$.p <- as.numeric(y_prob)
  tmp <- tmp[!is.na(tmp$.y) & !is.na(tmp$.p), ]
  if (nrow(tmp) == 0) return(data.frame())
  
  # quantile bins so each bin has similar size
  tmp$bin <- cut(tmp$.p,
                 breaks = unique(quantile(tmp$.p, probs = seq(0,1,length.out = n_bins+1), na.rm = TRUE)),
                 include.lowest = TRUE)
  
  tmp %>%
    group_by(bin) %>%
    summarise(
      n = n(),
      mean_pred = mean(.p),
      mean_obs  = mean(.y),
      .groups = "drop"
    )
}

plot_calibration <- function(cal_df, title = "Calibration plot") {
  if (nrow(cal_df) == 0) return(NULL)
  ggplot(cal_df, aes(x = mean_pred, y = mean_obs)) +
    geom_point() +
    geom_line() +
    geom_abline(intercept = 0, slope = 1, linetype = 2) +
    labs(x = "Mean predicted probability", y = "Observed event rate", title = title) +
    theme_minimal()
}
plot_calibration
# Calibration by subgroup (race, gender, age_bin etc)
calibration_by_group <- function(df, y_true, y_prob, subgroup_var, n_bins = 10) {
  tmp <- df
  tmp$.y <- as.numeric(y_true)
  tmp$.p <- as.numeric(y_prob)
  tmp$grp <- tmp[[subgroup_var]]
  tmp <- tmp[!is.na(tmp$.y) & !is.na(tmp$.p) & !is.na(tmp$grp) & as.character(tmp$grp)!="", ]
  if (nrow(tmp) == 0) return(data.frame())
  
  # compute calibration within each group
  out <- lapply(sort(unique(tmp$grp)), function(g) {
    d <- tmp[tmp$grp == g, ]
    if (length(unique(d$.y)) < 2) {
      # still allow table, but calibration is less meaningful if all outcomes same
      tab <- calibration_table(d, d$.y, d$.p, n_bins = n_bins)
      if (nrow(tab) == 0) return(NULL)
      tab$Group <- as.character(g)
      tab
    } else {
      tab <- calibration_table(d, d$.y, d$.p, n_bins = n_bins)
      if (nrow(tab) == 0) return(NULL)
      tab$Group <- as.character(g)
      tab
    }
  })
  out <- do.call(rbind, out)
  out
}

plot_calibration_groups <- function(cal_grp_df, title = "Calibration by group") {
  if (is.null(cal_grp_df) || nrow(cal_grp_df) == 0) return(NULL)
  ggplot(cal_grp_df, aes(x = mean_pred, y = mean_obs, color = Group)) +
    geom_point() +
    geom_line() +
    geom_abline(intercept = 0, slope = 1, linetype = 2) +
    labs(x = "Mean predicted probability", y = "Observed event rate", title = title) +
    theme_minimal()
}
# Calibration by subgroup (race, gender, age_bin etc)
calibration_by_group <- function(df, y_true, y_prob, subgroup_var, n_bins = 10) {
  tmp <- df
  tmp$.y <- as.numeric(y_true)
  tmp$.p <- as.numeric(y_prob)
  tmp$grp <- tmp[[subgroup_var]]
  tmp <- tmp[!is.na(tmp$.y) & !is.na(tmp$.p) & !is.na(tmp$grp) & as.character(tmp$grp)!="", ]
  if (nrow(tmp) == 0) return(data.frame())

  # compute calibration within each group
  out <- lapply(sort(unique(tmp$grp)), function(g) {
    d <- tmp[tmp$grp == g, ]
    if (length(unique(d$.y)) < 2) {
      # still allow table, but calibration is less meaningful if all outcomes same
      tab <- calibration_table(d, d$.y, d$.p, n_bins = n_bins)
      if (nrow(tab) == 0) return(NULL)
      tab$Group <- as.character(g)
      tab
    } else {
      tab <- calibration_table(d, d$.y, d$.p, n_bins = n_bins)
      if (nrow(tab) == 0) return(NULL)
      tab$Group <- as.character(g)
      tab
    }
  })
  out <- do.call(rbind, out)
  out
}

plot_calibration_groups <- function(cal_grp_df, title = "Calibration by group") {
  if (is.null(cal_grp_df) || nrow(cal_grp_df) == 0) return(NULL)
  ggplot(cal_grp_df, aes(x = mean_pred, y = mean_obs, color = Group)) +
    geom_point() +
    geom_line() +
    geom_abline(intercept = 0, slope = 1, linetype = 2) +
    labs(x = "Mean predicted probability", y = "Observed event rate", title = title) +
    theme_minimal()
}
# Example: Add age_bin to each test_use if not already present
add_age_bin <- function(df) {
  df$age_bin <- cut(df$age, c(0,39,64,200), labels = c("<40","40-64","65+"), include.lowest = TRUE)
  df
}
res_approached$test_use  <- add_age_bin(res_approached$test_use)
res_authorized$test_use  <- add_age_bin(res_authorized$test_use)
res_procured$test_use    <- add_age_bin(res_procured$test_use)

# Overall calibration + Brier score for each stage
brier_app  <- brier_score(res_approached$y_test, res_approached$prob_test)
brier_auth <- brier_score(res_authorized$y_test, res_authorized$prob_test)
brier_proc <- brier_score(res_procured$y_test,   res_procured$prob_test)

brier_app; brier_auth; brier_proc

cal_app  <- calibration_table(res_approached$test_use, res_approached$y_test, res_approached$prob_test, n_bins = 10)
cal_auth <- calibration_table(res_authorized$test_use, res_authorized$y_test, res_authorized$prob_test, n_bins = 10)
cal_proc <- calibration_table(res_procured$test_use,   res_procured$y_test,   res_procured$prob_test,   n_bins = 10)

p_cal_app  <- plot_calibration(cal_app,  "Calibration: Approached (overall)")
p_cal_auth <- plot_calibration(cal_auth, "Calibration: Authorized (overall)")
p_cal_proc <- plot_calibration(cal_proc, "Calibration: Procured (overall)")

print(p_cal_app); print(p_cal_auth); print(p_cal_proc)

# Subgroup calibration example: race
cal_app_race  <- calibration_by_group(res_approached$test_use, res_approached$y_test, res_approached$prob_test, "race", n_bins = 10)
cal_auth_race <- calibration_by_group(res_authorized$test_use, res_authorized$y_test, res_authorized$prob_test, "race", n_bins = 10)
cal_proc_race <- calibration_by_group(res_procured$test_use,   res_procured$y_test,   res_procured$prob_test,   "race", n_bins = 10)

p_cal_app_race  <- plot_calibration_groups(cal_app_race,  "Calibration by race: Approached")
p_cal_auth_race <- plot_calibration_groups(cal_auth_race, "Calibration by race: Authorized")
p_cal_proc_race <- plot_calibration_groups(cal_proc_race, "Calibration by race: Procured")

print(p_cal_app_race); print(p_cal_auth_race); print(p_cal_proc_race)

# You can repeat subgroup calibration with "gender" and "age_bin" similarly:
# calibration_by_group(..., subgroup_var="gender")
# calibration_by_group(..., subgroup_var="age_bin")
cal_app_gender  <- calibration_by_group(
  res_approached$test_use,
  res_approached$y_test,
  res_approached$prob_test,
  "gender",
  n_bins = 10
)

cal_auth_gender <- calibration_by_group(
  res_authorized$test_use,
  res_authorized$y_test,
  res_authorized$prob_test,
  "gender",
  n_bins = 10
)

cal_proc_gender <- calibration_by_group(
  res_procured$test_use,
  res_procured$y_test,
  res_procured$prob_test,
  "gender",
  n_bins = 10
)

p_cal_app_gender  <- plot_calibration_groups(cal_app_gender,  "Calibration by gender: Approached")
p_cal_auth_gender <- plot_calibration_groups(cal_auth_gender, "Calibration by gender: Authorized")
p_cal_proc_gender <- plot_calibration_groups(cal_proc_gender, "Calibration by gender: Procured")

print(p_cal_app_gender); print(p_cal_auth_gender); print(p_cal_proc_gender)

################################## Age Group############################
cal_app_age  <- calibration_by_group(
  res_approached$test_use,
  res_approached$y_test,
  res_approached$prob_test,
  "age_bin",
  n_bins = 10
)

cal_auth_age <- calibration_by_group(
  res_authorized$test_use,
  res_authorized$y_test,
  res_authorized$prob_test,
  "age_bin",
  n_bins = 10
)

cal_proc_age <- calibration_by_group(
  res_procured$test_use,
  res_procured$y_test,
  res_procured$prob_test,
  "age_bin",
  n_bins = 10
)

p_cal_app_age  <- plot_calibration_groups(cal_app_age,  "Calibration by age group: Approached")
p_cal_auth_age <- plot_calibration_groups(cal_auth_age, "Calibration by age group: Authorized")
p_cal_proc_age <- plot_calibration_groups(cal_proc_age, "Calibration by age group: Procured")

print(p_cal_app_age); print(p_cal_auth_age); print(p_cal_proc_age)


# -------------------------------------------------------
# 3) Fairness gap bar chart (race/gender/age across stages)
# -------------------------------------------------------
# Assumes you already have:
# fair_race_app, fair_gender_app, fair_age_app
# fair_race_authz, fair_gender_authz, fair_age_authz
# fair_race_proc, fair_gender_proc, fair_age_proc
# and your fairness_gaps() function

# If not, define fairness_gaps again safely
fairness_gaps <- function(results_df) {
  metrics <- c("Sensitivity","Specificity","Precision","PPR","AUC")
  gaps <- sapply(metrics, function(m) {
    max(results_df[[m]], na.rm = TRUE) - min(results_df[[m]], na.rm = TRUE)
  })
  data.frame(Metric = metrics, Gap = as.numeric(gaps))
}

make_gap_df <- function(gaps_df, stage, subgroup) {
  gaps_df %>%
    mutate(Stage = stage, Subgroup = subgroup)
}

gap_all <- bind_rows(
  make_gap_df(fairness_gaps(fair_race_app),   "Approached",  "Race"),
  make_gap_df(fairness_gaps(fair_gender_app), "Approached",  "Gender"),
  make_gap_df(fairness_gaps(fair_age_app),    "Approached",  "Age"),
  
  make_gap_df(fairness_gaps(fair_race_authz),   "Authorized", "Race"),
  make_gap_df(fairness_gaps(fair_gender_authz), "Authorized", "Gender"),
  make_gap_df(fairness_gaps(fair_age_authz),    "Authorized", "Age"),
  
  make_gap_df(fairness_gaps(fair_race_proc),   "Procured",   "Race"),
  make_gap_df(fairness_gaps(fair_gender_proc), "Procured",   "Gender"),
  make_gap_df(fairness_gaps(fair_age_proc),    "Procured",   "Age")
)

# Bar chart: gaps by Metric, grouped by Stage, faceted by Subgroup
p_gap_bar <- ggplot(gap_all, aes(x = Metric, y = Gap, fill = Stage)) +
  geom_col(position = position_dodge(width = 0.8)) +
  facet_wrap(~ Subgroup, nrow = 1) +
  labs(title = "Fairness gaps (max-min) across stages", x = "Metric", y = "Gap") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p_gap_bar)

# Save figure reliably
ggsave("fairness_gap_bar_chart.png", plot = p_gap_bar, width = 12, height = 4.5, dpi = 300)


# -------------------------------------------------------
# 4) Save calibration plots reliably (optional)
# -------------------------------------------------------
# Overall calibration
ggsave("calibration_overall_approached.png",  plot = p_cal_app,  width = 5.5, height = 4.5, dpi = 300)
ggsave("calibration_overall_authorized.png",  plot = p_cal_auth, width = 5.5, height = 4.5, dpi = 300)
ggsave("calibration_overall_procured.png",    plot = p_cal_proc, width = 5.5, height = 4.5, dpi = 300)

# By race (repeat similarly for gender/age_bin if you make those)
ggsave("calibration_race_approached.png", plot = p_cal_app_race,  width = 6.5, height = 5.0, dpi = 300)
ggsave("calibration_race_authorized.png", plot = p_cal_auth_race, width = 6.5, height = 5.0, dpi = 300)
ggsave("calibration_race_procured.png",   plot = p_cal_proc_race, width = 6.5, height = 5.0, dpi = 300)

# -------------------------------------------------------
# 5) Summary table (optional): AUC + AUPRC + Brier
# -------------------------------------------------------
perf_summary <- data.frame(
  Stage = c("Approached","Authorized","Procured"),
  AUROC = c(res_approached$auc, res_authorized$auc, res_procured$auc),
  AUPRC = c(res_approached$auprc, res_authorized$auprc, res_procured$auprc),
  Brier = c(brier_app, brier_auth, brier_proc)
)
perf_summary
write.csv(perf_summary, "xgb_performance_summary.csv", row.names = FALSE)
