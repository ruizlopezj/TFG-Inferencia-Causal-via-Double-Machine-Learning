#==============================================================================
# 1. INSTALACIÓN Y CARGA DE LIBRERÍAS
#==============================================================================
pkgs <- c(
  "dplyr", "readr", "ggplot2", "gridExtra", "corrplot", "reshape2",
  "DoubleML", "ranger", "mlr3", "mlr3learners", "data.table", "ggdag", "dagitty"
)

to_install <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(to_install) > 0) install.packages(to_install)

# 2. CARGA DE LIBRERÍAS
#------------------------------------------------------------------------------
library(dplyr)
library(readr)
library(ggplot2)
library(gridExtra)
library(corrplot)
library(reshape2)
library(sandwich)
library(lmtest)

library(ggplot2)     # ← ¡PARA labs() y ggdag!
library(ggdag)       # DAGs
library(dagitty)
library(dplyr)     # Para group_by/summarise  
library(readr)     # ← ¡PARA write_csv! 


# DAG causal
dag_tfg <- dagify(
  sales ~ price + temp + weekday + cost,  # Efectos directos + elast
  price ~ cost + weekday,                 # Endogeneidad
  exposure = "price",
  outcome = "sales",
  coords = list(x = c(temp=0, weekday=1, cost=2, price=3, sales=4),
                y = c(temp=1, weekday=1, cost=0, price=0.5, sales=0))
)

# Visualizamos el DAG que subyace de este ejemplo
ggdag(dag_tfg, text = TRUE) + 
  labs(title = "Grafo Acíclico Dirigido",
       subtitle = "precio endógeno: variables de confusión = {cost, weekday}")

# Aquí mostramos qué debemos controlar al ajustar el precio sobre las ventas 
adjustmentSets(dag_tfg, exposure = "price", outcome = "sales")


# Función mediante la cual generamos un conjunto de datos que nos otorgan una paradoja de Simpson
generar_datos<- function(n = 10000, seed = 5) {
  set.seed(seed)
  
  # Covariates (nodos raíz del DAG)
  temp <- round(rnorm(n, 24, 4), 1)
  
  weekday <- sample(1:7, n, replace = TRUE)
  
  cost <- sample(c(0.3, 0.5, 1.0, 1.5), n, replace = TRUE)
  
  # Nodo price (hijos del DAG)
  mu_price <- 5 + 0.4*cost + 1.0*(weekday == 6) + 0.6*(weekday %in% c(5,7)) + 0.08*temp
  
  price <- round(rnorm(n, mu_price, 1.2), 1)
  
  # Nodo sales (hoja del DAG, con tu elasticidad)
  elast <- -abs(-4.5 + 0.08*price + 0.02*temp + 
                  1.2*(weekday == 6) + 0.8*(weekday %in% c(5,7)) + 0.10*cost)
  
  mu_sales <- 170 + 45*(weekday == 6) + 32*(weekday %in% c(5,7)) + 
    12*(weekday %in% 1:4) + 5*temp + 6*cost + 3.5*elast*price
  
  sales <- as.integer(round(rnorm(n, mu_sales, 9)))
  
  data.frame(temp, weekday, cost, price, sales)
}

# Generación de datos
datos <- generar_datos(10000, 5)
write_csv(datos, "datos_helados.csv")

df_raw <- datos

head(df_raw)

colSums(is.na(df_raw))


# 3. AJUSTE DE TEMA GRÁFICO
#------------------------------------------------------------------------------
theme_set(
  theme_gray(base_family = "Times") +
    theme(
      axis.text = element_text(size = 10),
      axis.text.x = element_text(size = 10),
      axis.text.y = element_text(size = 10)
    )
)


#regresión lineal: A = intercept + beta_1*cost + beta_2*weekday + beta_3*temp
mod_p <- lm(price ~ cost + factor(weekday) + temp, data = df_raw)
#regresion lineal: Y = intercept + theta_1*cost + theta_2*weekday + theta_3*temp
mod_y <- lm(sales ~ cost + factor(weekday) + temp, data = df_raw)


df_residuos <- df_raw %>%
  mutate(
    resid_output_sales    = residuals(mod_y),
    resid_treatment_price = residuals(mod_p)
  )

df_plot <- bind_rows(
  df_residuos %>% filter(weekday %in% 1:4) %>% mutate(fase = "De lunes a jueves"),
  df_residuos %>% filter(weekday %in% 5:7) %>% mutate(fase = "Fin de semana"),
  df_residuos %>% mutate(fase = "Toda la semana")
)

df_plot$fase <- factor(df_plot$fase, levels = c("De lunes a jueves", "Fin de semana", "Toda la semana"))

#==============================================================================
# 4. VISUALIZACIONES (TRÍPTICOS)
#==============================================================================
cols <- colorRampPalette(c("green", "yellow", "red"))(7)
theme_tfg <- theme_minimal(base_size = 10) + theme(strip.background = element_rect(fill="gray95"))

# Generar y mostrar los 3 trípticos independientes
p1 <- ggplot(df_plot, aes(x = price, y = sales)) +
  geom_point(aes(color = factor(weekday)), alpha = 0.3) +
  geom_smooth(method = "lm", color = "black", se = FALSE) +
  facet_wrap(~fase) + scale_color_manual(values = cols, name = "Día") +
  labs(title = "Precio frente a ventas", x = "Precio", y = "Ventas") + theme_tfg

p2 <- ggplot(df_plot, aes(x = resid_treatment_price, y = sales)) +
  geom_point(aes(color = factor(weekday)), alpha = 0.3) +
  geom_smooth(method = "lm", color = "black", se = FALSE) +
  facet_wrap(~fase) + scale_color_manual(values = cols, name = "Día") +
  labs(title = "Ortogonalización del precio", x = "Residuos del precio", y = "Ventas") + theme_tfg

p3 <- ggplot(df_plot, aes(x = resid_treatment_price, y = resid_output_sales)) +
  geom_point(aes(color = factor(weekday)), alpha = 0.3) +
  geom_smooth(method = "lm", color = "black", se = FALSE) +
  facet_wrap(~fase) + scale_color_manual(values = cols, name = "Día") +
  labs(title = "Doble Ortogonalización (FWL)", x = "Residuos del precio", y = "Residuos de las ventas") + theme_tfg

print(p1); print(p2); print(p3)

#==============================================================================
# 5. COMPARATIVA FINAL
#==============================================================================
ate_naive <- coef(lm(sales ~ price, data = df_raw))[2]
ate_ols_ortho <- coef(lm(resid_output_sales ~ resid_treatment_price - 1, data = df_residuos))[1]

resumen <- data.frame(
  Metodo = c("Naive OLS", "FWL"),
  ATE = c(ate_naive, ate_ols_ortho)
)
print(resumen)