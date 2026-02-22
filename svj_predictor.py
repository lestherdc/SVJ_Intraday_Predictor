import yfinance as yf
import time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt


SYMBOL = "PLTR"
MARKET_REF = "^GSPC"
INTERVAL = "15m"
PERIOD = "60d"
NUM_PATHS = 30000
HORIZON_HOURS = 1
WINDOW_SIZE = 400
REFIT_INTERVAL_MIN = 15
DT = 15 / (60 * 6.5)

# Activar modo interactivo para que los gráficos no bloqueen el loop
plt.ion()


def clean_series(df):
    """Extrae y limpia la serie de precios detectando el tipo de objeto."""
    if df is None or (isinstance(df, (pd.DataFrame, pd.Series)) and df.empty):
        return np.array([])

    # Si es un DataFrame (posible MultiIndex de yfinance)
    if isinstance(df, pd.DataFrame):
        return df.iloc[:, 0].dropna().values.flatten().astype(float)
    # Si ya es una Serie de Pandas
    if isinstance(df, pd.Series):
        return df.dropna().values.flatten().astype(float)

    return np.array(df).flatten().astype(float)


def calculate_rsi(prices, period=14):
    if len(prices) < period + 1: return np.full(len(prices), 50.0)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period]) + 1e-10

    rsi = np.zeros_like(prices)
    rsi[:period] = 50.0
    for i in range(period, len(prices)):
        gain, loss = float(max(deltas[i - 1], 0)), float(max(-deltas[i - 1], 0))
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rsi[i] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    return rsi


def simulate_svj_with_beta(params, S0, v0, beta, market_drift, T, dt, n_paths):
    mu, kappa, theta, xi, rho, lambda_j, mu_j, sigma_j = params
    n_steps = 4
    dt_sim = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    v = np.full(n_paths, v0)

    for t in range(1, n_steps + 1):
        dz_s = np.random.normal(0, 1, n_paths)
        dz_v = np.random.normal(0, 1, n_paths)
        market_impact = beta * market_drift * dt_sim

        # Proceso de volatilidad
        v = np.maximum(v + kappa * (theta - v) * dt_sim + xi * np.sqrt(np.maximum(v, 0)) * (
                    rho * dz_s + np.sqrt(1 - rho ** 2) * dz_v), 1e-6)
        # Proceso de saltos
        jumps = np.random.poisson(max(0, lambda_j) * dt_sim, n_paths) * np.random.normal(mu_j, sigma_j, n_paths)

        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu + market_impact - 0.5 * v) * dt_sim + np.sqrt(v * dt_sim) * dz_s + jumps)
    return paths


# ======================
# INICIO DEL SCRIPT
# ======================
print(">>> SVJ Predictor <<<", flush=True)

try:
    while True:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Actualizando mercado...", flush=True)

        # Descargas
        raw_stock = yf.download(SYMBOL, interval=INTERVAL, period=PERIOD, progress=False)
        raw_market = yf.download(MARKET_REF, interval=INTERVAL, period=PERIOD, progress=False)

        closes_stock = clean_series(raw_stock['Close'])
        closes_market = clean_series(raw_market['Close'])

        if len(closes_stock) < 50:
            print("Datos insuficientes. El mercado podría estar cerrado o yfinance está fallando.", flush=True)
            time.sleep(60)
            continue

        # Análisis de retornos
        ret_stock = np.diff(np.log(closes_stock))
        ret_market = np.diff(np.log(closes_market))

        # Beta y Drift
        min_len = min(len(ret_stock), len(ret_market))
        beta = np.cov(ret_stock[-min_len:], ret_market[-min_len:])[0][1] / (np.var(ret_market[-min_len:]) + 1e-10)
        m_drift = np.mean(ret_market[-20:])

        m2 = np.var(ret_stock[-WINDOW_SIZE:])
        rsi_val = calculate_rsi(closes_stock)[-1]

        print(f"STATS -> Price: {closes_stock[-1]:.2f} | Beta: {beta:.2f} | RSI: {rsi_val:.2f}", flush=True)

        # Calibración
        bounds = [(-0.1, 0.1), (0.1, 3.0), (1e-4, 0.05), (0.01, 0.5), (-0.9, 0.0), (0.01, 1.0), (-0.1, 0.1),
                  (0.01, 0.1)]


        def obj(p):
            expected_var = (p[2] + p[5] * (p[6] ** 2 + p[7] ** 2)) * DT
            return (m2 - expected_var) ** 2


        result = differential_evolution(obj, bounds, maxiter=15, popsize=10)

        if result.success:
            p = result.x
            S0 = closes_stock[-1]
            paths = simulate_svj_with_beta(p, S0, m2 / DT, beta, m_drift, HORIZON_HOURS / 6.5, DT, NUM_PATHS)

            # Métricas
            p_up = np.mean(paths[:, -1] > S0)
            target = np.mean(paths[:, -1])

            print(f"PREDICCIÓN -> P(Subida): {p_up:.2%} | Objetivo 1h: ${target:.2f}", flush=True)

            # Gráfico rápido
            plt.clf()
            plt.style.use('dark_background')
            plt.plot(paths[:50].T, alpha=0.15, color='springgreen')
            plt.axhline(S0, color='white', linestyle='--', label='Actual')
            plt.title(f"{SYMBOL} Proyección - Prob: {p_up:.1%}")
            plt.draw()
            plt.pause(0.001)

        print(f"Próxima actualización en {REFIT_INTERVAL_MIN} min...", flush=True)
        time.sleep(REFIT_INTERVAL_MIN * 60)

except KeyboardInterrupt:
    print("\nCerrando sistema...")
except Exception as e:
    print(f"\nERROR CRÍTICO: {e}", flush=True)
    import traceback

    traceback.print_exc()