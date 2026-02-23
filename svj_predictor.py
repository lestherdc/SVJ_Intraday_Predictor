import yfinance as yf
import time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
np.random.seed(42)

SYMBOL = "PLTR"
MARKET_REF = "^GSPC"
INTERVAL = "15m"
PERIOD = "60d"
NUM_PATHS = 50000
HORIZON_HOURS = 1
WINDOW_SIZE = 400
REFIT_INTERVAL_MIN = 15
DT = 15 / (60 * 6.5)

plt.ion()


def clean_series(df):
    if df is None or (isinstance(df, (pd.DataFrame, pd.Series)) and df.empty):
        return np.array([])
    if isinstance(df, pd.DataFrame):
        return df.iloc[:, 0].dropna().values.flatten().astype(float)
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
    n_steps = 20
    dt_sim = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    v = np.full(n_paths, v0)
    for t in range(1, n_steps + 1):
        dz_s = np.random.normal(0, 1, n_paths)
        dz_v = np.random.normal(0, 1, n_paths)
        market_impact = beta * market_drift * dt_sim
        v = np.maximum(v + kappa * (theta - v) * dt_sim + xi * np.sqrt(np.maximum(v, 0)) * (
                    rho * dz_s + np.sqrt(1 - rho ** 2) * dz_v), 1e-6)
        jumps = np.random.poisson(max(0, lambda_j) * dt_sim, n_paths) * np.random.normal(mu_j, sigma_j, n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu + market_impact - 0.5 * v) * dt_sim + np.sqrt(v * dt_sim) * dz_s + jumps)
    return paths


# ======================
# BUCLE PRINCIPAL
# ======================
print(f">>> SISTEMA SVJ V3.3: MODO TRADING {SYMBOL} <<<", flush=True)

try:
    while True:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Escaneando mercado...", flush=True)

        # 1. Obtención de datos dual
        raw_stock = yf.download(SYMBOL, interval=INTERVAL, period=PERIOD, progress=False)
        raw_market = yf.download(MARKET_REF, interval=INTERVAL, period=PERIOD, progress=False)

        closes_stock = clean_series(raw_stock['Close'])
        closes_market = clean_series(raw_market['Close'])

        if len(closes_stock) < 50:
            print("Esperando datos de mercado abiertos...", flush=True)
            time.sleep(60)
            continue

        # 2. Análisis de Correlación y Momentum
        ret_stock = np.diff(np.log(closes_stock))
        ret_market = np.diff(np.log(closes_market))
        min_len = min(len(ret_stock), len(ret_market))
        beta = np.cov(ret_stock[-min_len:], ret_market[-min_len:])[0][1] / (np.var(ret_market[-min_len:]) + 1e-10)
        m_drift = np.mean(ret_market[-20:])
        m2 = np.var(ret_stock[-WINDOW_SIZE:])
        rsi_val = calculate_rsi(closes_stock)[-1]

        # 3. Calibración (Optimizador Evolutivo)
        bounds = [(-0.1, 0.1), (0.1, 3.0), (1e-4, 0.05), (0.01, 0.5), (-0.9, 0.0), (0.01, 1.0), (-0.1, 0.1),
                  (0.01, 0.1)]


        def obj(p):
            expected_var = (p[2] + p[5] * (p[6] ** 2 + p[7] ** 2)) * DT
            return (m2 - expected_var) ** 2


        result = differential_evolution(obj, bounds, maxiter=15, popsize=20, mutation=(0.5,1), recombination=0.7, strategy='best1bin', tol=0.001)

        #  Lógica de Índice de Confianza Normalizada, Health Check
        if True:  # Permitimos procesar para ver métricas siempre
            p = result.x

            # --- NUEVO: CÁLCULO SCI 100% ---
            error_factor = np.exp(-result.fun * 100)
            param_stability = 1.0 if (0.1 < p[1] < 10 and p[3] < 1.0) else 0.5
            market_context = 1.0 if (20 < rsi_val < 80) else 0.6

            confidence_index = (0.4 * error_factor + 0.3 * param_stability + 0.3 * market_context) * 100
            # -------------------------------

            S0 = closes_stock[-1]
            paths = simulate_svj_with_beta(p, S0, m2 / DT, beta, m_drift, HORIZON_HOURS / 6.5, DT, NUM_PATHS)

            p_up = np.mean(paths[:, -1] > S0)
            target = np.mean(paths[:, -1])
            status = "CONFIABLE" if confidence_index > 75 else "PRECAUCIÓN" if confidence_index > 50 else "NO FIABLE"

            print(f"STATS -> Price: {S0:.2f} | Beta: {beta:.2f} | RSI: {rsi_val:.2f}")
            print(f"SCI: {confidence_index:.1f}% [{status}]")
            print(f"PREDICCIÓN -> P(Subida): {p_up:.2%} | Obj: ${target:.2f}", flush=True)

            # Visualización interactiva
            plt.clf()
            #plt.style.use('dark_background')
            # Colorear según confianza
            #path_color = 'springgreen' if confidence_index > 70 else 'orange' if confidence_index > 40 else 'red'
            plt.plot(paths[:50].T, alpha=0.6, linewidth=0.8)
            plt.axhline(S0, color='red', linestyle='--', label=f'Precio Actual: {S0:.2f}')
            plt.title(f"{SYMBOL} (SCI: {confidence_index:.1f}%) - P(Up): {p_up:.1%}")
            plt.xlabel('Pasos (15min)')
            plt.ylabel('Precio')
            plt.legend()
            plt.draw()
            plt.pause(0.01)

        print(f"Dormiendo {REFIT_INTERVAL_MIN} min...", flush=True)
        time.sleep(REFIT_INTERVAL_MIN * 60)

except KeyboardInterrupt:
    print("\nSistema apagado por el usuario.")
except Exception as e:
    print(f"\nERROR: {e}")