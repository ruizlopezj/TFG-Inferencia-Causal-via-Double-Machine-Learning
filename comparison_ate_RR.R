############
#Simulacion de Montecarlo: Comparacion en terminos de sesgo y RR
# Metodos: G-computation, IPTW, AIPTW, DoubleML
# Escenarios: 1) Dual Misspecification, 2) Correctly Specified
############

rm(list = ls())

suppressPackageStartupMessages({
  library(DoubleML)
  library(mlr3)
  library(mlr3learners)
  library(data.table)
  library(ggplot2)
})

##############################################
# Opciones globales necesarias para que no
# explote en algunos casos el codigo
##############################################

lgr::get_logger("mlr3")$set_threshold("warn")
lgr::get_logger("bbotk")$set_threshold("warn")
options(warn = -1)

eps <- 1e-6

clip_prob <- function(p, eps = 1e-6) {
  pmin(pmax(p, eps), 1 - eps)
}

safe_mean <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  mean(x)
}

safe_abs_rel_bias <- function(est, truth) {
  est <- est[is.finite(est)]
  if (length(est) == 0 || !is.finite(truth) || abs(truth) < .Machine$double.eps) return(NA_real_)
  abs(mean((truth - est) / truth) * 100)
}

##############################################
# ESCENARIO 1: DUAL MISSPECIFICATION
##############################################

cat("\n========================================\n")
cat("ESCENARIO 1: DUAL MISSPECIFICATION\n")
cat("========================================\n")

set.seed(7777)

generateData_Misspec <- function(n) {
  w1 <- round(runif(n, min = 1, max = 5), digits = 0)
  w2 <- rbinom(n, size = 1, prob = 0.45)
  w3 <- round(runif(n, min = 0, max = 1), digits = 0 + 0.75 * w2 + 0.8 * w1)
  w4 <- round(runif(n, min = 0, max = 1), digits = 0 + 0.75 * w2 + 0.2 * w1)
  
  A <- rbinom(
    n,
    size = 1,
    prob = plogis(-1 - 0.15 * w4 + 1.5 * w2 + 0.75 * w3 + 0.25 * w1 + 0.8 * w2 * w4)
  )
  
  Y.1 <- rbinom(
    n,
    size = 1,
    prob = plogis(-3 + 1 + 0.25 * w4 + 0.75 * w3 + 0.8 * w2 * w4 + 0.05 * w1)
  )
  Y.0 <- rbinom(
    n,
    size = 1,
    prob = plogis(-3 + 0 + 0.25 * w4 + 0.75 * w3 + 0.8 * w2 * w4 + 0.05 * w1)
  )
  
  Y <- Y.1 * A + Y.0 * (1 - A)
  
  data.frame(w1, w2, w3, w4, A, Y, Y.1, Y.0)
}

ObsDataTrue_Misspec <- generateData_Misspec(n = 5000000)
True_ATE_Misspec <- mean(ObsDataTrue_Misspec$Y.1 - ObsDataTrue_Misspec$Y.0)
True_EY1_Misspec <- mean(ObsDataTrue_Misspec$Y.1)
True_EY0_Misspec <- mean(ObsDataTrue_Misspec$Y.0)
True_RR_Misspec <- True_EY1_Misspec / True_EY0_Misspec

cat("True ATE (Misspecified):", True_ATE_Misspec, "\n")
cat("True RR (Misspecified):", True_RR_Misspec, "\n\n")

##############################################
# ESCENARIO 2: CORRECTLY SPECIFIED
##############################################

cat("========================================\n")
cat("ESCENARIO 2: CORRECTLY SPECIFIED\n")
cat("========================================\n")

set.seed(8888)

generateData_Correct <- function(n) {
  w1 <- round(runif(n, min = 1, max = 5), digits = 0)
  w2 <- rbinom(n, size = 1, prob = 0.45)
  w3 <- round(runif(n, min = 0, max = 1), digits = 0 + 0.75 * w2 + 0.8 * w1)
  w4 <- round(runif(n, min = 0, max = 1), digits = 0 + 0.75 * w2 + 0.2 * w1)
  
  A <- rbinom(
    n,
    size = 1,
    prob = plogis(-1 - 0.15 * w4 + 1.5 * w2 + 0.75 * w3 + 0.25 * w1)
  )
  
  Y.1 <- rbinom(
    n,
    size = 1,
    prob = plogis(-3 + 1 + 0.25 * w4 + 0.75 * w3 + 0.05 * w1)
  )
  Y.0 <- rbinom(
    n,
    size = 1,
    prob = plogis(-3 + 0 + 0.25 * w4 + 0.75 * w3 + 0.05 * w1)
  )
  
  Y <- Y.1 * A + Y.0 * (1 - A)
  
  data.frame(w1, w2, w3, w4, A, Y, Y.1, Y.0)
}

ObsDataTrue_Correct <- generateData_Correct(n = 5000000)
True_ATE_Correct <- mean(ObsDataTrue_Correct$Y.1 - ObsDataTrue_Correct$Y.0)
True_EY1_Correct <- mean(ObsDataTrue_Correct$Y.1)
True_EY0_Correct <- mean(ObsDataTrue_Correct$Y.0)
True_RR_Correct <- True_EY1_Correct / True_EY0_Correct

cat("True ATE (Correct):", True_ATE_Correct, "\n")
cat("True RR (Correct):", True_RR_Correct, "\n\n")

##############################################
# FUNCION AUXILIAR PARA CALCULAR EL RR EN DML
##############################################

get_dml_rr <- function(SimData, n_folds = 5, eps = 1e-6, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  wvars <- c("w1", "w2", "w3", "w4")
  n <- nrow(SimData)
  
  folds <- sample(rep(1:n_folds, length.out = n))
  
  mu0_hat <- rep(NA_real_, n)
  mu1_hat <- rep(NA_real_, n)
  
  get_prob_one <- function(pred_obj) {
    p <- pred_obj$prob
    if ("1" %in% colnames(p)) return(as.numeric(p[, "1"]))
    if (ncol(p) == 2) return(as.numeric(p[, 1]))
    stop("Could not identify positive-class probability column.")
  }
  
  for (k in 1:n_folds) {
    test_id <- which(folds == k)
    train_id <- setdiff(seq_len(n), test_id)
    
    train_data <- SimData[train_id, , drop = FALSE]
    test_data  <- SimData[test_id, , drop = FALSE]
    
    train_0 <- train_data[train_data$A == 0, , drop = FALSE]
    train_1 <- train_data[train_data$A == 1, , drop = FALSE]
    
    # Si algún grupo queda vacío, devolvemos NA y dejamos que safe_mean gestione luego
    if (nrow(train_0) == 0 || nrow(train_1) == 0) next
    
    # Convertir Y a factor binario con niveles fijos para mlr3
    train_0$Y <- factor(train_0$Y, levels = c(0, 1))
    train_1$Y <- factor(train_1$Y, levels = c(0, 1))
    
    # Si solo hay una clase en el fold, usar predicción constante
    single_class_0 <- length(unique(train_0$Y)) < 2
    single_class_1 <- length(unique(train_1$Y)) < 2
    
    test_newdata <- test_data[, wvars, drop = FALSE]
    
    if (single_class_0) {
      mu0_hat[test_id] <- as.numeric(as.character(train_0$Y[1]))
    } else {
      learner_g0 <- lrn(
        "classif.ranger",
        predict_type = "prob",
        num.trees = 500,
        max.depth = 5,
        min.node.size = 2,
        num.threads = 1,
        verbose = FALSE
      )
      
      task_g0 <- TaskClassif$new(
        id = paste0("g0_", k),
        backend = train_0[, c(wvars, "Y"), drop = FALSE],
        target = "Y",
        positive = "1"
      )
      
      learner_g0$train(task_g0)
      mu0_hat[test_id] <- get_prob_one(learner_g0$predict_newdata(test_newdata))
    }
    
    if (single_class_1) {
      mu1_hat[test_id] <- as.numeric(as.character(train_1$Y[1]))
    } else {
      learner_g1 <- lrn(
        "classif.ranger",
        predict_type = "prob",
        num.trees = 500,
        max.depth = 5,
        min.node.size = 2,
        num.threads = 1,
        verbose = FALSE
      )
      
      task_g1 <- TaskClassif$new(
        id = paste0("g1_", k),
        backend = train_1[, c(wvars, "Y"), drop = FALSE],
        target = "Y",
        positive = "1"
      )
      
      learner_g1$train(task_g1)
      mu1_hat[test_id] <- get_prob_one(learner_g1$predict_newdata(test_newdata))
    }
  }
  
  mu0_hat <- clip_prob(mu0_hat, eps)
  mu1_hat <- clip_prob(mu1_hat, eps)
  
  ey1_hat <- safe_mean(mu1_hat)
  ey0_hat <- safe_mean(mu0_hat)
  
  rr_hat <- if (is.finite(ey1_hat) && is.finite(ey0_hat) && ey0_hat > eps) ey1_hat / ey0_hat else NA_real_
  
  list(
    EY1_DML = ey1_hat,
    EY0_DML = ey0_hat,
    RR_DML  = rr_hat,
    mu0_hat = mu0_hat,
    mu1_hat = mu1_hat
  )
}

##############################################
# FUNCION PARA LA SIMULACION
##############################################

run_simulation <- function(generateData_func, True_ATE, True_RR, scenario_name, R = 1000, n = 1000, eps = 1e-6) {
  
  cat("\n--- Running", R, "simulations for", scenario_name, "---\n")
  ATE_naif <- rep(NA_real_, R)
  RR_naif <- rep(NA_real_, R)
  ATE_gcomp <- rep(NA_real_, R)
  RR_gcomp <- rep(NA_real_, R)
  ATE_IPTW <- rep(NA_real_, R)
  RR_IPTW <- rep(NA_real_, R)
  ATE_AIPTW <- rep(NA_real_, R)
  RR_AIPTW <- rep(NA_real_, R)
  ATE_DML <- rep(NA_real_, R)
  RR_DML <- rep(NA_real_, R)
  
  for (r in 1:R) {
    if (r %% 100 == 0) cat("  Replication:", r, "/", R, "\n")
    
    SimData <- generateData_func(n = n)
    
    ##############################################
    # METODO 0: NAIF
    ##############################################
    
    # Subgrupos por tratamiento
    Y_treated   <- SimData$Y[SimData$A == 1]
    Y_control   <- SimData$Y[SimData$A == 0]
    
    EY1_naif <- safe_mean(Y_treated)
    EY0_naif <- safe_mean(Y_control)
    
    ATE_naif[r] <- EY1_naif - EY0_naif
    
    RR_naif[r]  <- if (is.finite(EY0_naif) && EY0_naif > eps) {
      EY1_naif / EY0_naif
    } else {
      NA_real_
    }
    
    ##############################################
    # METODO 1: G-COMPUTATION
    ##############################################
    
    suppressWarnings({
      gm <- glm(Y ~ A + w1 + w2 + w3 + w4, family = "binomial", data = SimData)
      
      newdata1 <- SimData[, c("w1", "w2", "w3", "w4")]
      newdata1$A <- 1
      newdata1 <- newdata1[, c("A", "w1", "w2", "w3", "w4")]
      
      newdata0 <- SimData[, c("w1", "w2", "w3", "w4")]
      newdata0$A <- 0
      newdata0 <- newdata0[, c("A", "w1", "w2", "w3", "w4")]
      
      Q1W <- clip_prob(predict(gm, newdata = newdata1, type = "response"), eps)
      Q0W <- clip_prob(predict(gm, newdata = newdata0, type = "response"), eps)
    })
    
    ATE_gcomp[r] <- safe_mean(Q1W - Q0W)
    RR_gcomp[r]  <- {
      m1 <- safe_mean(Q1W)
      m0 <- safe_mean(Q0W)
      if (is.finite(m1) && is.finite(m0) && m0 > eps) m1 / m0 else NA_real_
    }
    
    ##############################################
    # METODO 2: IPTW
    ##############################################
    
    suppressWarnings({
      psm <- glm(A ~ w1 + w2 + w3 + w4, family = binomial, data = SimData)
      gW <- clip_prob(predict(psm, type = "response"), eps)
    })
    
    num1 <- sum(SimData$A * SimData$Y / gW)
    den1 <- sum(SimData$A / gW)
    num0 <- sum((1 - SimData$A) * SimData$Y / (1 - gW))
    den0 <- sum((1 - SimData$A) / (1 - gW))
    
    EY1_IPTW <- if (is.finite(num1) && is.finite(den1) && den1 > eps) num1 / den1 else NA_real_
    EY0_IPTW <- if (is.finite(num0) && is.finite(den0) && den0 > eps) num0 / den0 else NA_real_
    
    ATE_IPTW[r] <- if (is.finite(EY1_IPTW) && is.finite(EY0_IPTW)) EY1_IPTW - EY0_IPTW else NA_real_
    RR_IPTW[r]  <- if (is.finite(EY1_IPTW) && is.finite(EY0_IPTW) && EY0_IPTW > eps) EY1_IPTW / EY0_IPTW else NA_real_
    
    ##############################################
    # METODO 3: AIPTW
    ##############################################
    
    # Modelo de resultado: predicciones contrafactuales bajo tratamiento y control.
    Q1W <- clip_prob(predict(gm, newdata = data.frame(
      A = 1,
      SimData[, c("w1", "w2", "w3", "w4")]
    ), type = "response"), eps)
    
    Q0W <- clip_prob(predict(gm, newdata = data.frame(
      A = 0,
      SimData[, c("w1", "w2", "w3", "w4")]
    ), type = "response"), eps)
    
    # Estimación AIPTW de E[Y(1)] y E[Y(0)].
    mu1_AIPTW <- safe_mean(SimData$A * (SimData$Y - Q1W) / gW + Q1W)
    mu0_AIPTW <- safe_mean((1 - SimData$A) * (SimData$Y - Q0W) / (1 - gW) + Q0W)
    
    # ATE como diferencia de medias potenciales estimadas.
    ATE_AIPTW[r] <- if (is.finite(mu1_AIPTW) && is.finite(mu0_AIPTW)) mu1_AIPTW - mu0_AIPTW else NA_real_
    
    # RR como cociente de las medias potenciales estimadas.
    RR_AIPTW[r] <- if (is.finite(mu1_AIPTW) && is.finite(mu0_AIPTW) && mu0_AIPTW > eps) mu1_AIPTW / mu0_AIPTW else NA_real_
    
    
    ##############################################
    # METODO 4: DoubleML
    ##############################################
    
    suppressMessages(suppressWarnings({
      dml_data <- data.table(
        Y = SimData$Y,
        D = SimData$A,
        w1 = SimData$w1,
        w2 = SimData$w2,
        w3 = SimData$w3,
        w4 = SimData$w4
      )
      
      obj_dml_data <- DoubleMLData$new(
        dml_data,
        y_col = "Y",
        d_cols = "D",
        x_cols = c("w1", "w2", "w3", "w4")
      )
      
      learner_rf <- lrn(
        "classif.ranger",
        predict_type = "prob",
        num.trees = 500,
        max.depth = 5,
        min.node.size = 2,
        num.threads = 1,
        verbose = FALSE
      )
      
      dml_irm <- DoubleMLIRM$new(
        obj_dml_data,
        ml_g = learner_rf,
        ml_m = learner_rf,
        n_folds = 5,
        score = "ATE"
      )
      
      dml_irm$fit()
      ATE_DML[r] <- as.numeric(dml_irm$coef)
      
      rr_dml_obj <- get_dml_rr(SimData, n_folds = 5, eps = eps, seed = 1000 + r)
      RR_DML[r] <- rr_dml_obj$RR_DML
    }))
  }
  
  ##############################################
  # CALCULO DE BIAS
  ##############################################
  bias_naif_ATE  <- safe_abs_rel_bias(ATE_naif,  True_ATE)
  bias_gcomp_ATE <- safe_abs_rel_bias(ATE_gcomp, True_ATE)
  bias_IPTW_ATE   <- safe_abs_rel_bias(ATE_IPTW, True_ATE)
  bias_AIPTW_ATE <- safe_abs_rel_bias(ATE_AIPTW, True_ATE)
  bias_DML_ATE   <- safe_abs_rel_bias(ATE_DML, True_ATE)
  
  bias_naif_RR   <- safe_abs_rel_bias(RR_naif,   True_RR)
  bias_gcomp_RR  <- safe_abs_rel_bias(RR_gcomp, True_RR)
  bias_IPTW_RR    <- safe_abs_rel_bias(RR_IPTW, True_RR)
  bias_AIPTW_RR  <- safe_abs_rel_bias(RR_AIPTW, True_RR)
  bias_DML_RR    <- safe_abs_rel_bias(RR_DML, True_RR)
  
  results <- data.frame(
    Scenario = scenario_name,
    Method = c("Naif", "G-computation", "IPTW", "AIPTW", "DoubleML"),
    Mean_ATE = c(safe_mean(ATE_naif), safe_mean(ATE_gcomp), safe_mean(ATE_IPTW), safe_mean(ATE_AIPTW), safe_mean(ATE_DML)),
    Bias_ATE_pct = c( bias_naif_ATE, bias_gcomp_ATE, bias_IPTW_ATE, bias_AIPTW_ATE, bias_DML_ATE),
    Mean_RR = c(  safe_mean(RR_naif), safe_mean(RR_gcomp), safe_mean(RR_IPTW), safe_mean(RR_AIPTW), safe_mean(RR_DML)),
    Bias_RR_pct = c( bias_naif_RR,  bias_gcomp_RR, bias_IPTW_RR, bias_AIPTW_RR, bias_DML_RR)
  )
  
  return(list(
    results = results,
    ATE_naif = ATE_naif,
    ATE_gcomp = ATE_gcomp,
    ATE_IPTW = ATE_IPTW,
    ATE_AIPTW = ATE_AIPTW,
    ATE_DML = ATE_DML,
    RR_naif = RR_naif,
    RR_gcomp = RR_gcomp,
    RR_IPTW = RR_IPTW,
    RR_AIPTW = RR_AIPTW,
    RR_DML = RR_DML
  ))
}

##############################################
# COMPILAMOS AMBOS ESCENARIOS
##############################################

results_misspec <- run_simulation(
  generateData_func = generateData_Misspec,
  True_ATE = True_ATE_Misspec,
  True_RR = True_RR_Misspec,
  scenario_name = "Dual Misspecification",
  R = 1000,
  n = 1000,
  eps = eps
)

results_correct <- run_simulation(
  generateData_func = generateData_Correct,
  True_ATE = True_ATE_Correct,
  True_RR = True_RR_Correct,
  scenario_name = "Correctly Specified",
  R = 1000,
  n = 1000,
  eps = eps
)

##############################################
# MOSTRAMOS LOS RESULTADOS
##############################################

cat("\n\n========================================\n")
cat("FINAL RESULTS COMPARISON\n")
cat("========================================\n\n")

cat("--- SCENARIO 1: DUAL MISSPECIFICATION ---\n")
print(results_misspec$results)

cat("\n--- SCENARIO 2: CORRECTLY SPECIFIED ---\n")
print(results_correct$results)

combined_results <- rbind(results_misspec$results, results_correct$results)
cat("\n--- COMBINED RESULTS ---\n")
print(combined_results)

##############################################
# VISUALIZACION DE LOS RESULTADOS
##############################################

plot_data <- combined_results[is.finite(combined_results$Bias_ATE_pct), ]

p1 <- ggplot(plot_data, aes(x = Method, y = Bias_ATE_pct, fill = Scenario)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(
    title = "Sesgo relativo en la estimación del ATE (%)",
    y = "Sesgo relativo en valor absoluto (%)",
    x = "Método"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p1)
ggsave("bias_comparison_ATE.png", p1, width = 10, height = 6)

plot_data_RR <- combined_results[is.finite(combined_results$Bias_RR_pct), ]

p2 <- ggplot(plot_data_RR, aes(x = Method, y = Bias_RR_pct, fill = Scenario)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(
    title = "Sesgo relativo en la estimación del RR (%)",
    y = "Sesgo relativo en valor absoluto (%)",
    x = "Método"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p2)
ggsave("bias_comparison_RR.png", p2, width = 10, height = 6)

##############################################
# GUARDAMOS LOS RESULTADOS
##############################################

save(
  results_misspec, results_correct, combined_results,
  True_ATE_Misspec, True_RR_Misspec,
  True_ATE_Correct, True_RR_Correct,
  file = "simulation_results_DoubleML.RData"
)

cat("\n\nResults saved to: simulation_results_DoubleML.RData\n")
cat("Plots saved to: bias_comparison_ATE.png and bias_comparison_RR.png\n")

options(warn = 0)