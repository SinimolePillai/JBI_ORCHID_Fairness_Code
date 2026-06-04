library(pROC)
library(caret)
library(dplyr)
library(ggplot2)
library(PRROC)

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
#####################################################

fit_logit_binary <- function(train_df, test_df, outcome_var, feature_vars) {
  
  # Ensure complete cases
  keep_tr <- complete.cases(train_df[, c(feature_vars, outcome_var)])
  train_use <- train_df[keep_tr, ]
  
  keep_te <- complete.cases(test_df[, c(feature_vars, outcome_var)])
  test_use <- test_df[keep_te, ]
  
  # Outcome
  y_train <- as.numeric(train_use[[outcome_var]])
  y_test  <- as.numeric(test_use[[outcome_var]])
  
  # Formula (factors handled automatically)
  fml <- as.formula(
    paste(outcome_var, "~", paste(feature_vars, collapse = " + "))
  )
  
  # Fit logistic regression
  model <- glm(
    fml,
    data = train_use,
    family = binomial(link = "logit")
  )
  
  # Predicted probabilities
  prob <- predict(model, newdata = test_use, type = "response")
  
  # AUC
  roc_obj <- roc(y_test, prob, quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  
  # Best threshold (Youden)
  coords_best <- coords(
    roc_obj, x = "best", best.method = "youden",
    ret = c("threshold","sensitivity","specificity"),
    transpose = FALSE
  )
  best_thresh <- as.numeric(coords_best["threshold"])
  
  # Class predictions
  pred <- ifelse(prob >= best_thresh, 1, 0)
  
  # Confusion matrix
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
    test_use = test_use
  )
}


logit_approached <- fit_logit_binary(train_df_base, test_df_base, "approached", feature_vars2)
logit_authorized <- fit_logit_binary(train_df_base, test_df_base, "authorized", feature_vars2)
logit_procured   <- fit_logit_binary(train_df_base, test_df_base, "procured",   feature_vars2)

logit_approached$auc
logit_authorized$auc
logit_procured$auc

pr_auc <- function(y_true, y_prob) {
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

logit_approached$auprc <- pr_auc(logit_approached$y_test, logit_approached$prob_test)
logit_authorized$auprc <- pr_auc(logit_authorized$y_test, logit_authorized$prob_test)
logit_procured$auprc   <- pr_auc(logit_procured$y_test,   logit_procured$prob_test)

add_age_bin <- function(df) {
  df$age_bin <- cut(df$age, c(0,39,64,200),
                    labels = c("<40","40-64","65+"),
                    include.lowest = TRUE)
  df
}

logit_approached$test_use <- add_age_bin(logit_approached$test_use)
logit_authorized$test_use <- add_age_bin(logit_authorized$test_use)
logit_procured$test_use   <- add_age_bin(logit_procured$test_use)
#########################################################################
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


fair_race_app_logit   <- fairness_metrics_test(logit_approached$test_use,
                                               logit_approached$y_test,
                                               logit_approached$prob_test,
                                               logit_approached$pred_test,
                                               "race")

fair_gender_app_logit <- fairness_metrics_test(logit_approached$test_use,
                                               logit_approached$y_test,
                                               logit_approached$prob_test,
                                               logit_approached$pred_test,
                                               "gender")

fair_age_app_logit    <- fairness_metrics_test(logit_approached$test_use,
                                               logit_approached$y_test,
                                               logit_approached$prob_test,
                                               logit_approached$pred_test,
                                               "age_bin")
####################################
# AUTHORIZED — Logistic regression

fair_race_auth_logit <- fairness_metrics_test(
  logit_authorized$test_use,
  logit_authorized$y_test,
  logit_authorized$prob_test,
  logit_authorized$pred_test,
  "race"
)

fair_gender_auth_logit <- fairness_metrics_test(
  logit_authorized$test_use,
  logit_authorized$y_test,
  logit_authorized$prob_test,
  logit_authorized$pred_test,
  "gender"
)

fair_age_auth_logit <- fairness_metrics_test(
  logit_authorized$test_use,
  logit_authorized$y_test,
  logit_authorized$prob_test,
  logit_authorized$pred_test,
  "age_bin"
)
fairness_gaps <- function(fair_df) {
  metrics <- c("Sensitivity","Specificity","Precision","PPR","AUC")
  bind_rows(lapply(metrics, function(m) {
    data.frame(
      Metric = m,
      Gap = max(fair_df[[m]], na.rm = TRUE) -
        min(fair_df[[m]], na.rm = TRUE)
    )
  }))
}

fairness_gaps(fair_race_auth_logit)
fairness_gaps(fair_gender_auth_logit)
fairness_gaps(fair_age_auth_logit)

# PROCURED — Logistic regression

fair_race_proc_logit <- fairness_metrics_test(
  logit_procured$test_use,
  logit_procured$y_test,
  logit_procured$prob_test,
  logit_procured$pred_test,
  "race"
)

fair_gender_proc_logit <- fairness_metrics_test(
  logit_procured$test_use,
  logit_procured$y_test,
  logit_procured$prob_test,
  logit_procured$pred_test,
  "gender"
)

fair_age_proc_logit <- fairness_metrics_test(
  logit_procured$test_use,
  logit_procured$y_test,
  logit_procured$prob_test,
  logit_procured$pred_test,
  "age_bin"
)

fairness_gaps(fair_race_proc_logit)
fairness_gaps(fair_gender_proc_logit)
fairness_gaps(fair_age_proc_logit)
################################################
make_gap_df <- function(gaps_df, stage, subgroup, model) {
  gaps_df %>%
    mutate(Stage = stage,
           Subgroup = subgroup,
           Model = model)
}
gap_logit_all <- bind_rows(
  # APPROACHED
  make_gap_df(fairness_gaps(fair_race_app_logit),   "Approached", "Race",   "Logistic"),
  make_gap_df(fairness_gaps(fair_gender_app_logit), "Approached", "Gender", "Logistic"),
  make_gap_df(fairness_gaps(fair_age_app_logit),    "Approached", "Age",    "Logistic"),
  
  # AUTHORIZED
  make_gap_df(fairness_gaps(fair_race_auth_logit),   "Authorized", "Race",   "Logistic"),
  make_gap_df(fairness_gaps(fair_gender_auth_logit), "Authorized", "Gender", "Logistic"),
  make_gap_df(fairness_gaps(fair_age_auth_logit),    "Authorized", "Age",    "Logistic"),
  
  # PROCURED
  make_gap_df(fairness_gaps(fair_race_proc_logit),   "Procured", "Race",   "Logistic"),
  make_gap_df(fairness_gaps(fair_gender_proc_logit), "Procured", "Gender", "Logistic"),
  make_gap_df(fairness_gaps(fair_age_proc_logit),    "Procured", "Age",    "Logistic")
)
p_gap_bar_logit <- ggplot(
  gap_logit_all,
  aes(x = Metric, y = Gap, fill = Stage)
) +
  geom_col(position = position_dodge(width = 0.8)) +
  facet_wrap(~ Subgroup, nrow = 1) +
  labs(
    title = "Fairness gaps (Logistic Regression)",
    x = "Metric",
    y = "Gap (max − min)"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p_gap_bar_logit)
###########################################################################
#####################Calibration#############################
# Overall calibration — Logistic


calibration_table <- function(test_df, y_true, y_prob, n_bins = 10) {
  df <- data.frame(y = y_true, p = y_prob)
  df$bin <- cut(df$p,
                breaks = quantile(df$p, probs = seq(0,1,length.out = n_bins+1),
                                  na.rm = TRUE),
                include.lowest = TRUE)
  df %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(p, na.rm = TRUE),
      mean_obs  = mean(y, na.rm = TRUE),
      n = n()
    )
}


plot_calibration <- function(cal_df, title) {
  ggplot(cal_df, aes(mean_pred, mean_obs)) +
    geom_point() +
    geom_line() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(title = title, x = "Predicted", y = "Observed") +
    theme_minimal()
}

calibration_by_group <- function(test_df, y_true, y_prob, group_var, n_bins = 10) {
  bind_rows(lapply(split(seq_len(nrow(test_df)), test_df[[group_var]]), function(idx) {
    g <- as.character(test_df[[group_var]][idx[1]])
    cal <- calibration_table(test_df[idx,], y_true[idx], y_prob[idx], n_bins)
    cal$Group <- g
    cal
  }))
}

plot_calibration_groups <- function(cal_df, title) {
  ggplot(cal_df, aes(mean_pred, mean_obs, colour = Group)) +
    geom_point() +
    geom_line() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(title = title, x = "Predicted", y = "Observed") +
    theme_minimal()
}
cal_app_logit  <- calibration_table(
  logit_approached$test_use,
  logit_approached$y_test,
  logit_approached$prob_test,
  n_bins = 10
)

cal_auth_logit <- calibration_table(
  logit_authorized$test_use,
  logit_authorized$y_test,
  logit_authorized$prob_test,
  n_bins = 10
)

cal_proc_logit <- calibration_table(
  logit_procured$test_use,
  logit_procured$y_test,
  logit_procured$prob_test,
  n_bins = 10
)

p_cal_app_logit  <- plot_calibration(cal_app_logit,  "Calibration: Approached (Logistic)")
p_cal_auth_logit <- plot_calibration(cal_auth_logit, "Calibration: Authorized (Logistic)")
p_cal_proc_logit <- plot_calibration(cal_proc_logit, "Calibration: Procured (Logistic)")

print(p_cal_app_logit); print(p_cal_auth_logit); print(p_cal_proc_logit)
########################################



cal_app_logit  <- calibration_table(logit_approached$test_use,
                                    logit_approached$y_test,
                                    logit_approached$prob_test)

p_cal_app_logit <- plot_calibration(cal_app_logit,
                                    "Calibration: Approached (Logistic)")
print(p_cal_app_logit)

cal_app_race_logit <- calibration_by_group(
  logit_approached$test_use,
  logit_approached$y_test,
  logit_approached$prob_test,
  "race",
  n_bins = 10
)
cal_app_gender_logit <- calibration_by_group(
  logit_approached$test_use,
  logit_approached$y_test,
  logit_approached$prob_test,
  "gender",
  n_bins = 10
)
p_cal_app_gender_logit <- plot_calibration_groups(
  cal_app_gender_logit,
  "Calibration by gender: Approached (Logistic)"
)

print(p_cal_app_race_logit)
print(p_cal_app_gender_logit)
print(p_cal_app_race_logit)

brier_score <- function(y_true, y_prob) {
  y_true <- as.numeric(y_true)
  mean((y_prob - y_true)^2, na.rm = TRUE)
}

brier_app_logit  <- brier_score(logit_approached$y_test, logit_approached$prob_test)
brier_auth_logit <- brier_score(logit_authorized$y_test, logit_authorized$prob_test)
brier_proc_logit <- brier_score(logit_procured$y_test,   logit_procured$prob_test)
perf_compare <- data.frame(
  Stage = c("Approached","Authorized","Procured"),
  
  AUROC_XGB = c(res_approached$auc, res_authorized$auc, res_procured$auc),
  AUROC_LOG = c(logit_approached$auc, logit_authorized$auc, logit_procured$auc),
  
  AUPRC_XGB = c(res_approached$auprc, res_authorized$auprc, res_procured$auprc),
  AUPRC_LOG = c(logit_approached$auprc, logit_authorized$auprc, logit_procured$auprc),
  
  Brier_XGB = c(brier_app, brier_auth, brier_proc),
  Brier_LOG = c(brier_app_logit, brier_auth_logit, brier_proc_logit)
)

perf_compare
write.csv(perf_compare, "model_comparison_xgb_vs_logit.csv", row.names = FALSE)
