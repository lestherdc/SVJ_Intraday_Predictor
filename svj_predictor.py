import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Configuración
SYMBOL = "TSLA"  # Cambia a "AAPL" o "PLTR"
INTERVAL = "15m"
PERIOD = "60d"  # Máximo para intradía en yfinance
NUM_PATHS = 10000  # Para probabilidades
HORIZON_HOURS = 1  # Predicción para próxima hora
STEPS_PER_HOUR = 4  # 15min steps (60/15=4)

# Descargar datos
df = yf.download(SYMBOL, interval=INTERVAL, period=PERIOD)
closes = df['Close'].values


# Función para simular paths SVJ (Euler discretización)
def simulate_svj(params, S0, v0, T, dt, n_paths):
    mu, kappa, theta, xi, rho, lambda_j, mu_j, sigma_j = params
    n_steps = int(T / dt)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    v = np.full(n_paths, v0)

    for t in range(1, n_steps + 1):
        dW_s = np.random.normal(0, np.sqrt(dt), n_paths)
        dW_v = rho * dW_s + np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), n_paths)
        v = np.maximum(v + kappa * (theta - v) * dt + xi * np.sqrt(v) * dW_v, 0)
        jumps = np.random.poisson(lambda_j * dt, n_paths) * np.random.normal(mu_j, sigma_j, n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp((mu - 0.5 * v) * dt + np.sqrt(v * dt) * dW_s + jumps)

    return paths


# Función de likelihood para calibración (approximate MLE con moments)
def svj_log_likelihood(params, returns, dt):
    mu, kappa, theta, xi, rho, lambda_j, mu_j, sigma_j = params
    mean_ret = np.mean(returns)
    var_ret = np.var(returns)
    skew_ret = np.mean((returns - mean_ret) ** 3) / var_ret ** 1.5 if var_ret > 0 else 0
    kurt_ret = np.mean((returns - mean_ret) ** 4) / var_ret ** 2 - 3 if var_ret > 0 else 0

    # Moments esperados aproximados bajo SVJ
    expected_mean = (mu - lambda_j * mu_j) * dt
    expected_var = theta * dt + lambda_j * (sigma_j ** 2 + mu_j ** 2) * dt
    expected_skew = lambda_j * (
                3 * sigma_j ** 2 * mu_j + mu_j ** 3) * dt / expected_var ** 1.5 if expected_var > 0 else 0
    expected_kurt = lambda_j * (
                3 * sigma_j ** 4 + 6 * sigma_j ** 2 * mu_j ** 2 + mu_j ** 4) * dt / expected_var ** 2 if expected_var > 0 else 0

    # Loss como suma de diferencias cuadradas
    loss = (mean_ret - expected_mean) ** 2 + (var_ret - expected_var) ** 2 + (skew_ret - expected_skew) ** 2 + (
                kurt_ret - expected_kurt) ** 2
    return loss


# Preparar returns y dt
log_returns = np.log(closes[1:] / closes[:-1])
dt = 15 / (60 * 390)  # 15min fracción de trading day (390min/día)

# Params iniciales: mu, kappa, theta, xi, rho, lambda_j, mu_j, sigma_j
initial_params = [0.0, 2.0, 0.04, 0.3, -0.5, 0.5, -0.03, 0.1]

# Optimizar calibración
result = minimize(svj_log_likelihood, initial_params, args=(log_returns, dt), method='Nelder-Mead',
                  bounds=[(-1, 1), (0.1, 10), (0.01, 0.2), (0.01, 1), (-1, 1), (0.01, 2), (-0.5, 0.5), (0.01, 0.5)])
calibrated_params = result.x
print("Parámetros calibrados:", calibrated_params)
print("Éxito optimización:", result.success)
print("Loss final:", result.fun)

# Simular para próxima hora
S0 = closes[-1]
v0 = np.var(log_returns[-100:]) if len(log_returns) > 100 else 0.04  # Estimado vol reciente
T = HORIZON_HOURS / 6.5  # Hora en fracción de trading day (6.5h día)
dt = T / STEPS_PER_HOUR  # Steps de 15min

paths = simulate_svj(calibrated_params, S0, v0, T, dt, NUM_PATHS)

# Calcular probabilidades y stats
final_prices = paths[:, -1]
p_up = np.mean(final_prices > S0)
p_up_1pct = np.mean(final_prices > S0 * 1.01)
mean_pred = np.mean(final_prices)
ci_90_low = np.percentile(final_prices, 5)
ci_90_high = np.percentile(final_prices, 95)

print(f"\nPredicciones para próxima hora (basado en última vela):")
print(f"P(subida): {p_up:.2%}")
print(f"P(subida >1%): {p_up_1pct:.2%}")
print(f"Precio medio esperado: ${mean_pred:.2f}")
print(f"90% Intervalo de confianza: [${ci_90_low:.2f}, ${ci_90_high:.2f}]")

# Plot 100 paths sample
plt.figure(figsize=(10, 6))
plt.plot(paths[:100].T)
plt.title(f'Sample SVJ Paths para {SYMBOL} Próxima Hora (15min steps)')
plt.xlabel('Steps')
plt.ylabel('Precio')
plt.show()