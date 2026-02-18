import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Configuración
SYMBOL = "TSLA"  # Cambia a "AAPL" o "PLTR" si quieres
INTERVAL = "15m"
PERIOD = "60d"  # Máximo para intradía en yfinance
NUM_PATHS = 10000  # Número de simulaciones Monte Carlo
HORIZON_HOURS = 1  # Predicción para próxima hora
STEPS_PER_HOUR = 4  # 15min steps (60/15 = 4)

# Descargar datos
print(f"Descargando {SYMBOL} en {INTERVAL}...")
df = yf.download(SYMBOL, interval=INTERVAL, period=PERIOD, progress=False)

if df.empty:
    print("Error: No se descargaron datos. Verifica conexión o símbolo.")
    exit()

# Extraer como arrays NumPy con dtype explícito float64
closes = df['Close'].values.astype(np.float64)
volumes = df['Volume'].values.astype(np.float64)

# Normalizar volumen
if len(volumes) > 100:
    vol_mean = np.mean(volumes[-100:])
    vol_std = np.std(volumes[-100:])
else:
    vol_mean = np.mean(volumes)
    vol_std = np.std(volumes)

normalized_vol = (volumes - vol_mean) / (vol_std + 1e-8)

# Imprimir último precio usando .item() para extraer escalar seguro
print(f"Datos obtenidos: {len(closes)} velas")
print(f"Último precio: ${closes[-1].item():.2f}")


# Función para simular paths SVJ
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


# Función de likelihood con penalización por volumen
def svj_log_likelihood(params, returns, dt, norm_vol):
    mu, kappa, theta, xi, rho, lambda_j, mu_j, sigma_j = params

    mean_ret = np.mean(returns)
    var_ret = np.var(returns)
    skew_ret = np.mean((returns - mean_ret) ** 3) / var_ret ** 1.5 if var_ret > 0 else 0
    kurt_ret = np.mean((returns - mean_ret) ** 4) / var_ret ** 2 - 3 if var_ret > 0 else 0

    expected_mean = (mu - lambda_j * mu_j) * dt
    expected_var = theta * dt + lambda_j * (sigma_j ** 2 + mu_j ** 2) * dt
    expected_skew = lambda_j * (
                3 * sigma_j ** 2 * mu_j + mu_j ** 3) * dt / expected_var ** 1.5 if expected_var > 0 else 0
    expected_kurt = lambda_j * (
                3 * sigma_j ** 4 + 6 * sigma_j ** 2 * mu_j ** 2 + mu_j ** 4) * dt / expected_var ** 2 if expected_var > 0 else 0

    vol_divergence = np.mean(np.abs(norm_vol[-len(returns):]))
    volume_penalty = vol_divergence * 0.5

    loss = ((mean_ret - expected_mean) ** 2 + (var_ret - expected_var) ** 2 +
            (skew_ret - expected_skew) ** 2 + (kurt_ret - expected_kurt) ** 2 + volume_penalty)

    return loss


# Preparar returns y dt
log_returns = np.log(closes[1:] / closes[:-1])
dt = 15 / (60 * 390)  # 15 min como fracción de trading day

# Parámetros iniciales
initial_params = [0.0, 2.0, 0.04, 0.3, -0.5, 0.5, -0.03, 0.1]

# Calibración
print("Calibrando parámetros SVJ con volumen...")
result = minimize(svj_log_likelihood, initial_params,
                  args=(log_returns, dt, normalized_vol),
                  method='Nelder-Mead',
                  bounds=[(-1, 1), (0.1, 10), (0.01, 0.2), (0.01, 1), (-1, 1), (0.01, 2), (-0.5, 0.5), (0.01, 0.5)])

calibrated_params = result.x
print("Parámetros calibrados:", calibrated_params)
print("Éxito optimización:", result.success)
print("Loss final:", result.fun)

# Simular
S0 = closes[-1]
v0 = np.var(log_returns[-100:]) if len(log_returns) > 100 else 0.04

T = HORIZON_HOURS / 6.5
dt_sim = T / STEPS_PER_HOUR

paths = simulate_svj(calibrated_params, S0, v0, T, dt_sim, NUM_PATHS)

# Resultados
final_prices = paths[:, -1]
p_up = np.mean(final_prices > S0)
p_up_1pct = np.mean(final_prices > S0 * 1.01)
mean_pred = np.mean(final_prices)
ci_90_low = np.percentile(final_prices, 5)
ci_90_high = np.percentile(final_prices, 95)

print(f"\nPredicciones para próxima hora ({SYMBOL}):")
print(f"P(subida): {p_up:.2%}")
print(f"P(subida >1%): {p_up_1pct:.2%}")
print(f"Precio medio esperado: ${mean_pred:.2f}")
print(f"90% Intervalo de confianza: [${ci_90_low:.2f}, ${ci_90_high:.2f}]")

# Gráfico
plt.figure(figsize=(10, 6))
plt.plot(paths[:100].T, alpha=0.6, linewidth=0.8)
plt.axhline(S0, color='red', linestyle='--', label=f'Precio actual: ${S0.item():.2f}')
plt.title(f'SVJ + Volumen: {NUM_PATHS} paths - Próxima hora ({SYMBOL})')
plt.xlabel('Pasos (15 min)')
plt.ylabel('Precio')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()