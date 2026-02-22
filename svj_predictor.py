import yfinance as yf
import time
from datetime import datetime
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Configuración General
SYMBOL = "PLTR"  # Cambia a "AAPL" o "PLTR" si quieres
INTERVAL = "15m"
PERIOD = "60d"
OPTIMIZER_METHOD = "L-BFGS-B"  # L-BFGS-B o Nelder-Mead
NUM_PATHS = 100000  # Alta precisión
HORIZON_HOURS = 1
STEPS_PER_HOUR = 4

# Configuración real-time
REFIT_INTERVAL_MIN = 15  # recalibra cada 15 min
SHOW_EVERY_REFIT = True  # Muestra predicciones cada vez

# dt e initial_params definidos al inicio
dt = 15 / (60 * 390)  # fracción de día para 15 min
initial_params = [0.0, 2.0, 0.04, 0.3, -0.5, 0.5, -0.03, 0.1]

# Función RSI manual robusta
def calculate_rsi(prices, period=14):
    prices = np.array(prices, dtype=float)
    n = len(prices)

    rsi = np.full(n, 50.0)

    if n <= period:
        return rsi

    deltas = np.diff(prices)
    if len(deltas) == 0:
        return rsi

    seed = deltas[:period]
    if seed.size == 0:
        return rsi

    gains = np.maximum(seed, 0)
    losses = np.abs(np.minimum(seed, 0))

    avg_gain = np.mean(gains) if gains.size > 0 else 0.0
    avg_loss = np.mean(losses) if losses.size > 0 else 1e-10

    if avg_loss > 0:
        rs = avg_gain / avg_loss
        rsi[period] = 100. - 100. / (1. + rs)

    for i in range(period + 1, n):
        delta = deltas[i - 1]
        gain = max(delta, 0)
        loss = max(-delta, 0)

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100. - 100. / (1. + rs)

    return rsi

# Función simulación SVJ
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

# Likelihood con volumen + RSI + VIX
def svj_log_likelihood(params, returns, dt, norm_vol, norm_rsi, norm_vix):
    mu, kappa, theta, xi, rho, lambda_j, mu_j, sigma_j = params

    mean_ret = np.mean(returns)
    var_ret = np.var(returns)
    skew_ret = np.mean((returns - mean_ret) ** 3) / var_ret ** 1.5 if var_ret > 0 else 0
    kurt_ret = np.mean((returns - mean_ret) ** 4) / var_ret ** 2 - 3 if var_ret > 0 else 0

    expected_mean = (mu - lambda_j * mu_j) * dt
    expected_var = theta * dt + lambda_j * (sigma_j ** 2 + mu_j ** 2) * dt
    expected_skew = lambda_j * (3 * sigma_j ** 2 * mu_j + mu_j ** 3) * dt / expected_var ** 1.5 if expected_var > 0 else 0
    expected_kurt = lambda_j * (3 * sigma_j ** 4 + 6 * sigma_j ** 2 * mu_j ** 2 + mu_j ** 4) * dt / expected_var ** 2 if expected_var > 0 else 0

    vol_divergence = np.mean(np.abs(norm_vol[-len(returns):]))
    volume_penalty = vol_divergence * 0.5

    rsi_extreme = np.mean(np.abs(norm_rsi[-len(returns):]))
    rsi_penalty = rsi_extreme * 0.4

    vix_divergence = np.mean(np.abs(norm_vix[-len(returns):]))
    vix_penalty = vix_divergence * 0.2

    loss = ((mean_ret - expected_mean) ** 2 + (var_ret - expected_var) ** 2 +
            (skew_ret - expected_skew) ** 2 + (kurt_ret - expected_kurt) ** 2 +
            volume_penalty + rsi_penalty + vix_penalty)

    return loss

# ======================
# MODO REAL-TIME
# ======================
print(f"\n=== MODO REAL-TIME ACTIVADO ===")
print(f"Recalibrando cada {REFIT_INTERVAL_MIN} minutos. Presiona Ctrl+C para detener.")

try:
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{current_time}] Iniciando recalibración...")

        # Descargar datos frescos
        df = yf.download(SYMBOL, interval=INTERVAL, period=PERIOD, progress=False)
        if df.empty:
            print("Error: No se descargaron datos. Esperando próxima iteración...")
            time.sleep(REFIT_INTERVAL_MIN * 60)
            continue

        vix_df = yf.download("^VIX", interval=INTERVAL, period=PERIOD, progress=False)
        if vix_df.empty:
            vix_closes = np.full(len(df), 20.0)
        else:
            vix_closes = vix_df['Close'].values.astype(np.float64)
            if len(vix_closes) < len(df):
                vix_closes = np.pad(vix_closes, (len(df) - len(vix_closes), 0), mode='edge')

        closes = df['Close'].values.astype(np.float64)
        volumes = df['Volume'].values.astype(np.float64)

        rsi_values = calculate_rsi(closes)
        normalized_rsi = (rsi_values - 50) / 50
        if len(normalized_rsi) > len(closes) - 1:
            normalized_rsi = normalized_rsi[:len(closes)-1]
        else:
            pad_length = len(closes) - 1 - len(normalized_rsi)
            normalized_rsi = np.pad(normalized_rsi, (pad_length, 0), mode='edge')

        # Rolling window: últimas 500 velas
        window_size = 500
        if len(closes) > window_size:
            closes_window = closes[-window_size:]
            volumes_window = volumes[-window_size:]
            rsi_window = rsi_values[-window_size:]
            vix_window = vix_closes[-window_size:]
        else:
            closes_window = closes
            volumes_window = volumes
            rsi_window = rsi_values
            vix_window = vix_closes

        # Normalizar dentro del window
        vol_mean = np.mean(volumes_window[-100:]) if len(volumes_window) > 100 else np.mean(volumes_window)
        vol_std = np.std(volumes_window[-100:]) if len(volumes_window) > 100 else np.std(volumes_window)
        normalized_vol_window = (volumes_window - vol_mean) / (vol_std + 1e-8)

        vix_mean = np.mean(vix_window[-100:]) if len(vix_window) > 100 else np.mean(vix_window)
        vix_std = np.std(vix_window[-100:]) if len(vix_window) > 100 else np.std(vix_window)
        normalized_vix_window = (vix_window - vix_mean) / (vix_std + 1e-8)

        # Preparar log_returns del window
        closes_array_window = np.array(closes_window)
        log_returns_window = np.log(closes_array_window[1:] / closes_array_window[:-1])
        normalized_rsi_window = normalized_rsi[-len(log_returns_window):]
        normalized_vol_window = normalized_vol_window[-len(log_returns_window):]
        normalized_vix_window = normalized_vix_window[-len(log_returns_window):]

        # Calibrar
        print(f"Calibrando con {len(log_returns_window)} velas recientes...")
        result = minimize(svj_log_likelihood, initial_params,
                          args=(log_returns_window, dt, normalized_vol_window, normalized_rsi_window, normalized_vix_window),
                          method=OPTIMIZER_METHOD,
                          bounds=[(-1,1), (0.1,10), (0.01,0.2), (0.01,1), (-1,1), (0.01,2), (-0.5,0.5), (0.01,0.5)])

        if result.success:
            calibrated_params = result.x
            print("Parámetros recalibrados:", calibrated_params)
            print("Loss final:", result.fun)

            # Simular
            S0 = closes[-1]
            v0 = np.var(log_returns_window[-100:]) if len(log_returns_window) > 100 else 0.04
            T = HORIZON_HOURS / 6.5
            dt_sim = T / STEPS_PER_HOUR
            paths = simulate_svj(calibrated_params, S0, v0, T, dt_sim, NUM_PATHS)

            final_prices = paths[:, -1]
            p_up = np.mean(final_prices > S0)
            p_up_1pct = np.mean(final_prices > S0 * 1.01)
            mean_pred = np.mean(final_prices)
            lower_bound = mean_pred * 0.99
            upper_bound = mean_pred * 1.01
            p_near_mean = np.mean((final_prices >= lower_bound) & (final_prices <= upper_bound))
            ci_90_low = np.percentile(final_prices, 5)
            ci_90_high = np.percentile(final_prices, 95)

            print(f"\nPredicciones actualizadas para próximas {HORIZON_HOURS} horas ({SYMBOL}):")
            print(f"P(subida): {p_up:.2%}")
            print(f"P(subida >1%): {p_up_1pct:.2%}")
            print(f"Precio medio esperado: ${mean_pred:.2f}")
            print(f"Probabilidad de estar dentro de ±1% del precio medio esperado: {p_near_mean:.2%}")
            print(f"90% Intervalo de confianza: [${ci_90_low:.2f}, ${ci_90_high:.2f}]")

            # Gráfico actualizado cada recalibración
            plt.figure(figsize=(10,6))
            plt.plot(paths[:100].T, alpha=0.6, linewidth=0.8)
            plt.axhline(S0, color='red', linestyle='--', label=f'Precio actual: ${S0.item():.2f}')
            plt.title(f'SVJ + Volumen + RSI + VIX ({current_time}) - Próximas {HORIZON_HOURS} horas ({SYMBOL})')
            plt.xlabel('Pasos (15 min)')
            plt.ylabel('Precio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            plt.close()  # Cierra la ventana para no acumular gráficos
        else:
            print("No se pudo recalibrar. Usando parámetros anteriores.")

        print(f"Esperando {REFIT_INTERVAL_MIN} minutos para próxima actualización...\n")
        time.sleep(REFIT_INTERVAL_MIN * 60)

except KeyboardInterrupt:
    print("\nReal-time detenido por el usuario. ¡Hasta la próxima!")
except Exception as e:
    print(f"Error inesperado en real-time: {e}")