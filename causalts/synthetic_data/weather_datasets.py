# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Synthetic ERCOT-like weather and generation benchmark dataset.

Generates 6-hourly wind speed (m/s) and cloud cover (%) for 8 diverse
Texas stations over one year, calibrated to ERCOT actuals (wind speed
distributions, diurnal/seasonal patterns, cross-station correlations)
and publicly available ERCOT generation statistics.

Values are stochastic — not copies of real data — and reproducible via seed.

Causal structure encoded in ground_truth:
- Wind self-AR at lag 1 and 2 (all stations)
- Cloud self-AR at lag 1 (all stations)
- Spatial wind→wind at lag 1 (south-to-north prevailing flow: 6 directed edges)
- Same-station wind→cloud at lag 0 (contemporaneous, negative: windier = less cloud)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Station metadata (calibrated from ERCOT actuals & ECMWF ensemble mean)
# ---------------------------------------------------------------------------

STATIONS = ["kcrp", "kdfw", "kelp", "kama", "kiah", "kabi", "ksps", "kmaf"]

STATION_META: dict[str, dict] = {
    # Corpus Christi — south coast, Gulf of Mexico
    "kcrp": {
        "lat": 27.77,
        "lon": -97.50,
        "wind_mean": 4.96,
        "wind_std": 2.76,
        "cloud_mean": 47.4,
        "A_diurnal": 0.22,
        "A_seasonal": 0.17,
    },
    # Dallas-Fort Worth — north central
    "kdfw": {
        "lat": 32.90,
        "lon": -97.04,
        "wind_mean": 4.62,
        "wind_std": 2.47,
        "cloud_mean": 42.2,
        "A_diurnal": 0.18,
        "A_seasonal": 0.14,
    },
    # El Paso — far west, isolated
    "kelp": {
        "lat": 31.81,
        "lon": -106.38,
        "wind_mean": 3.62,
        "wind_std": 2.57,
        "cloud_mean": 30.8,
        "A_diurnal": 0.16,
        "A_seasonal": 0.12,
    },
    # Amarillo — panhandle, highest wind
    "kama": {
        "lat": 35.22,
        "lon": -101.71,
        "wind_mean": 5.82,
        "wind_std": 2.78,
        "cloud_mean": 39.3,
        "A_diurnal": 0.20,
        "A_seasonal": 0.16,
    },
    # Houston — southeast, Gulf coast
    "kiah": {
        "lat": 29.98,
        "lon": -95.34,
        "wind_mean": 3.46,
        "wind_std": 2.14,
        "cloud_mean": 48.8,
        "A_diurnal": 0.17,
        "A_seasonal": 0.13,
    },
    # Abilene — west central
    "kabi": {
        "lat": 32.41,
        "lon": -99.68,
        "wind_mean": 4.66,
        "wind_std": 2.45,
        "cloud_mean": 39.1,
        "A_diurnal": 0.19,
        "A_seasonal": 0.14,
    },
    # Wichita Falls — north
    "ksps": {
        "lat": 33.98,
        "lon": -98.49,
        "wind_mean": 4.89,
        "wind_std": 2.60,
        "cloud_mean": 40.1,
        "A_diurnal": 0.18,
        "A_seasonal": 0.14,
    },
    # Midland-Odessa — Permian Basin
    "kmaf": {
        "lat": 31.94,
        "lon": -102.20,
        "wind_mean": 4.89,
        "wind_std": 2.53,
        "cloud_mean": 36.7,
        "A_diurnal": 0.18,
        "A_seasonal": 0.14,
    },
}

# Spatial wind→wind causal edges at lag 1 (prevailing SSE/south→north flow)
# Format: (cause_station, effect_station, coefficient)
_SPATIAL_EDGES: list[tuple[str, str, float]] = [
    ("kcrp", "kiah", 0.13),  # Corpus Christi → Houston (coastal flow)
    ("kiah", "kdfw", 0.10),  # Houston → Dallas (north)
    ("kdfw", "ksps", 0.11),  # Dallas → Wichita Falls (further north)
    ("kama", "ksps", 0.10),  # Amarillo → Wichita Falls (panhandle → north)
    ("kelp", "kmaf", 0.12),  # El Paso → Midland (west → Permian Basin)
    ("kmaf", "kabi", 0.10),  # Midland → Abilene (Permian → central)
]

# Weibull shape fixed at k=2 (Rayleigh; canonical for wind; empirically k≈2.07)
_WEIBULL_K = 2.0
# Γ(1 + 1/k) for k=2 → Γ(1.5) = √π/2 ≈ 0.8862
_WEIBULL_MEAN_FACTOR = 0.8862

# VAR coefficients for log(scale) in Weibull model
_AR1_DIAG = 0.50  # lag-1 self-persistence (calibrated from actuals AR(1)≈0.50)
_AR2_DIAG = 0.25  # lag-2 self-persistence

# Cloud logit-space AR coefficient
_CLOUD_RHO = 0.60
# Wind→cloud contemporaneous coefficient (logit space, negative = windier→clearer)
_WIND_CLOUD_GAMMA = -0.05
# Logit-space cloud innovation std
_CLOUD_SIGMA_Z = 0.45

# Solar constants
_S0 = 1361.0  # W/m² solar constant
# Cloud attenuation: GHI = TOA * (1 - 0.75 * (cloud/100)^3.4)
_CLOUD_ATTN_A = 0.75
_CLOUD_ATTN_B = 3.4

# Capacity weights per station — normalized fractions of ERCOT installed capacity.
# Wind: dominated by West Texas (KMAF), Abilene area (KABI), and Panhandle (KAMA).
# El Paso (KELP) is on the El Paso Electric grid, not ERCOT — zero weight.
# Sources: ERCOT wind/solar zones, major farm locations (Horse Hollow/Sweetwater ~KABI,
# Permian Basin farms ~KMAF, Panhandle hubs ~KAMA/KSPS, South Texas ~KCRP).
WIND_CAPACITY_WEIGHTS: dict[str, float] = {
    "kcrp": 0.11,  # South Texas zone, some coastal farms
    "kdfw": 0.06,  # DFW area, minimal wind
    "kelp": 0.00,  # El Paso Electric grid — outside ERCOT
    "kama": 0.22,  # Panhandle hub, major wind zone
    "kiah": 0.06,  # Houston area, minimal wind
    "kabi": 0.20,  # Taylor/Nolan counties — Horse Hollow, Sweetwater, Buffalo Gap
    "ksps": 0.11,  # North Texas, secondary Panhandle
    "kmaf": 0.24,  # Permian Basin — largest single ERCOT wind cluster
}
# Solar: West Texas (KMAF) and South Texas (KCRP) lead due to irradiance + development.
# DFW and Houston have growing utility-scale solar near load centers.
SOLAR_CAPACITY_WEIGHTS: dict[str, float] = {
    "kcrp": 0.30,  # South Texas, high irradiance + active solar development
    "kdfw": 0.14,  # DFW metro fringe, growing utility solar
    "kelp": 0.00,  # Outside ERCOT
    "kama": 0.03,  # Panhandle, good irradiance but less developed
    "kiah": 0.14,  # Houston area, moderate solar
    "kabi": 0.11,  # West-central, emerging solar zone (Taylor County)
    "ksps": 0.03,  # North Texas, limited solar build-out
    "kmaf": 0.25,  # Permian Basin, highest irradiance + rapid solar deployment
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _wind_log_scale_mean(
    timestamps: pd.DatetimeIndex,
    meta: dict,
    stationary_var_r: float = 0.0,
) -> np.ndarray:
    """Time-varying log(scale) mean for the Weibull model. Shape: (T,).

    Applies Jensen's correction (-stationary_var_r/2) so that E[W_i] equals
    the calibration target despite the log-normal inflation from VAR residuals.
    """
    # Jensen correction: E[exp(r)] = exp(σ²_r/2) > 1, so subtract σ²_r/2
    base_log = np.log(meta["wind_mean"] / _WEIBULL_MEAN_FACTOR) - stationary_var_r / 2
    hour = timestamps.hour.to_numpy(dtype=float)
    doy = timestamps.day_of_year.to_numpy(dtype=float)
    # Diurnal: trough at noon (hour=12 UTC ≈ local morning in Texas; the harmonic
    # matches the local-time pattern via the hour variable which is UTC).
    # We shift diurnal trough to UTC 18 (≈ local noon in Texas, UTC-6).
    diurnal = meta["A_diurnal"] * np.cos(2 * np.pi * (hour - 18) / 24)
    # Seasonal: peak around day 100 (≈ April 10), trough in Sept
    seasonal = meta["A_seasonal"] * np.cos(2 * np.pi * (doy - 100) / 365)
    return base_log + diurnal + seasonal


def _build_wind_var_matrices(n: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build B1, B2 (VAR coefficient matrices) and Sigma (noise covariance)."""
    B1 = np.zeros((n, n))
    B2 = np.zeros((n, n))

    np.fill_diagonal(B1, _AR1_DIAG)
    np.fill_diagonal(B2, _AR2_DIAG)

    for cause, effect, coef in _SPATIAL_EDGES:
        ci = STATIONS.index(cause)
        ei = STATIONS.index(effect)
        B1[ci, ei] = coef

    # Residual noise std: σ² ≈ Var(log λ) * (1 - AR1² - AR2²)
    # Calibrate so that Var(log W) matches empirical std (roughly)
    # log(W) std ≈ log(1 + (std/mean)²)^0.5 per station
    sigma_diag = np.array(
        [
            np.sqrt(
                np.log(
                    1
                    + (STATION_META[s]["wind_std"] / STATION_META[s]["wind_mean"]) ** 2
                )
            )
            * np.sqrt(max(0.01, 1 - _AR1_DIAG**2 - _AR2_DIAG**2))
            for s in STATIONS
        ]
    )

    # Residual cross-correlation (small, geographically decaying)
    # Approximate: coastal pair (kcrp-kiah) 0.15, north cluster 0.10, others 0.05
    corr_pairs: dict[tuple[int, int], float] = {
        (STATIONS.index("kcrp"), STATIONS.index("kiah")): 0.15,
        (
            STATIONS.index("kiah"),
            STATIONS.index("kbpt") if "kbpt" in STATIONS else 0,
        ): 0.0,
        (STATIONS.index("kdfw"), STATIONS.index("ksps")): 0.10,
        (STATIONS.index("kama"), STATIONS.index("ksps")): 0.08,
        (STATIONS.index("kabi"), STATIONS.index("kmaf")): 0.08,
    }
    corr_mat = np.eye(n)
    for (i, j), c in corr_pairs.items():
        if 0 <= i < n and 0 <= j < n and i != j:
            corr_mat[i, j] = corr_mat[j, i] = c

    Sigma = np.outer(sigma_diag, sigma_diag) * corr_mat
    # Ensure positive definite via small ridge
    Sigma += np.eye(n) * 1e-6
    return B1, B2, Sigma


def _simulate_wind(
    rng: np.random.Generator,
    T: int,
    B1: np.ndarray,
    B2: np.ndarray,
    Sigma: np.ndarray,
    mu_log: np.ndarray,  # shape (T, 8)
) -> np.ndarray:
    """Simulate wind speed via Weibull(k=2, scale=exp(log_lambda)).

    log_lambda follows a VAR(2) process around the seasonal+diurnal mean.
    Returns W of shape (T, 8) in m/s.
    """
    n = len(STATIONS)
    L = np.linalg.cholesky(Sigma)
    burn = 200

    r = np.zeros((T + burn, n))
    for t in range(2, T + burn):
        eps = L @ rng.standard_normal(n)
        r[t] = B1.T @ r[t - 1] + B2.T @ r[t - 2] + eps

    r = r[burn:]  # (T, n)

    # log(scale) = mean + residual
    log_lam = mu_log + r  # (T, n)
    lam = np.exp(log_lam)  # Weibull scale

    # W ~ Weibull(k=2, scale=lam): W = lam * (-log(U))^(1/k)  for U ~ Uniform(0,1)
    u = rng.uniform(size=(T, n))
    W = lam * (-np.log(np.clip(u, 1e-12, 1 - 1e-12))) ** (1.0 / _WEIBULL_K)
    # Cap at physical maximum observed in ERCOT actuals (24.6 m/s) + headroom
    return np.clip(W, 0.0, 30.0)


def _toa_instantaneous(
    doy: np.ndarray,  # (T, 1) fractional day of year
    hour_utc: np.ndarray,  # (T, 1) fractional UTC hour
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Instantaneous TOA irradiance. Shape: (T, n_stations)."""
    B = 2 * np.pi * (doy - 1) / 365
    ecc = 1.00011 + 0.034221 * np.cos(B) + 0.001280 * np.sin(B)
    delta = np.radians(23.45 * np.sin(2 * np.pi * (doy - 81) / 365))
    hour_solar = hour_utc + lons[None, :] / 15.0
    ha = np.radians((hour_solar - 12.0) * 15.0)
    lat_rad = np.radians(lats)[None, :]
    cos_theta = np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(
        delta
    ) * np.cos(ha)
    return _S0 * ecc * np.clip(cos_theta, 0.0, 1.0)


def _toa_irradiance(
    timestamps: pd.DatetimeIndex,
    lats: np.ndarray,
    lons: np.ndarray,
    n_sub: int = 12,
) -> np.ndarray:
    """Mean TOA irradiance averaged over each 6-hour window. Shape: (T, n_stations).

    Samples the instantaneous TOA formula at n_sub equally-spaced offsets within
    each 6-hour window and averages. This prevents the sawtooth artifact that
    arises from point-sampling at 4 timestamps per day, giving a smooth daily arch
    that correctly represents the energy received during each interval.
    """
    T = len(timestamps)
    result = np.zeros((T, len(lats)))
    step_hours = 6.0 / n_sub

    for k in range(n_sub):
        offset_min = k * step_hours * 60
        t_off = timestamps + pd.Timedelta(minutes=offset_min)
        doy = t_off.day_of_year.to_numpy(dtype=float)[:, None]
        hour_utc = (t_off.hour + t_off.minute / 60.0).to_numpy(dtype=float)[:, None]
        result += _toa_instantaneous(doy, hour_utc, lats, lons)

    return result / n_sub


def _simulate_cloud(
    rng: np.random.Generator,
    T: int,
    cloud_means: np.ndarray,  # (n,) in percent
    W: np.ndarray,  # (T, n) wind speed
) -> np.ndarray:
    """Simulate cloud cover in logit space with AR(1) and wind coupling.

    Returns C of shape (T, n) in percent [0, 100].
    """
    n = len(STATIONS)
    burn = 100

    # Logit of mean cloud cover per station
    mu_z = np.log(cloud_means / (100.0 - cloud_means))  # (n,)

    # Correlated noise: small cross-station correlation
    corr_cloud = np.full((n, n), 0.05)
    np.fill_diagonal(corr_cloud, 1.0)
    # Coastal pair more correlated
    i_kcrp, i_kiah = STATIONS.index("kcrp"), STATIONS.index("kiah")
    corr_cloud[i_kcrp, i_kiah] = corr_cloud[i_kiah, i_kcrp] = 0.15
    Sig_z = corr_cloud * _CLOUD_SIGMA_Z**2
    Sig_z += np.eye(n) * 1e-6
    L_z = np.linalg.cholesky(Sig_z)

    # Wind is pre-simulated; pad with zeros for burn period
    W_padded = np.concatenate([np.ones((burn, n)) * W[:burn].mean(axis=0), W], axis=0)

    z = np.zeros((T + burn, n))
    z[0] = mu_z
    for t in range(1, T + burn):
        eps_z = L_z @ rng.standard_normal(n)
        # Normalise wind effect: use log(W+1) to reduce leverage of high winds
        wind_effect = _WIND_CLOUD_GAMMA * np.log1p(W_padded[t])
        z[t] = mu_z + _CLOUD_RHO * (z[t - 1] - mu_z) + wind_effect + eps_z

    z = z[burn:]  # (T, n)
    C = 100.0 / (1.0 + np.exp(-z))  # sigmoid → (0, 100)
    return np.clip(C, 0.0, 100.0)


def _stationary_var_lyapunov(
    B1: np.ndarray, B2: np.ndarray, Sigma: np.ndarray
) -> np.ndarray:
    """Exact stationary variance of r via discrete Lyapunov on VAR(2) companion.

    Companion form: x(t) = A x(t-1) + e(t)
    A = [[B1^T, B2^T], [I, 0]]  — stacked state [r(t), r(t-1)]
    Q = [[Sigma, 0], [0, 0]]     — companion noise covariance

    Solves P = A P A^T + Q for P, returns diag(P[:n, :n]).
    Falls back to an iterative estimate if scipy is unavailable.
    """
    from scipy.linalg import solve_discrete_lyapunov

    n = B1.shape[0]
    A = np.block([[B1.T, B2.T], [np.eye(n), np.zeros((n, n))]])
    Q = np.zeros((2 * n, 2 * n))
    Q[:n, :n] = Sigma
    P = solve_discrete_lyapunov(A, Q)
    return np.diag(P[:n, :n])


def _power_curve(
    v: np.ndarray,
    v_ci: float = 3.0,
    v_r: float = 12.0,
    v_sd: float = 20.0,
    v_co: float = 25.0,
) -> np.ndarray:
    """Wind turbine power curve → capacity factor in [0, 1].

    Four regions:
    - v < v_ci           : 0 (below cut-in)
    - v_ci ≤ v ≤ v_r    : cubic ramp (IEC standard)
    - v_r < v ≤ v_sd    : 1.0 (rated power)
    - v_sd < v ≤ v_co   : smooth cosine decline from 1 → 0 (soft cut-out)
    - v > v_co           : 0

    The soft cut-out region (v_sd=20 m/s → v_co=25 m/s) reflects fleet-level
    behavior: individual turbines begin feathering at different speeds so
    aggregated generation tapers smoothly rather than cliffing at a single speed.
    """
    denom = v_r**3 - v_ci**3
    # Cosine taper: 1 at v_sd, 0 at v_co
    taper = 0.5 * (1.0 + np.cos(np.pi * (v - v_sd) / (v_co - v_sd)))
    cf = np.where(
        v < v_ci,
        0.0,
        np.where(
            v <= v_r,
            (v**3 - v_ci**3) / denom,
            np.where(
                v <= v_sd,
                1.0,
                np.where(v <= v_co, taper, 0.0),
            ),
        ),
    )
    return cf


def _build_ground_truth() -> np.ndarray:
    """Binary causal adjacency tensor, shape (16, 16, 3) for max_lag=2."""
    n_stations = len(STATIONS)
    d = 2 * n_stations  # 16 variables
    G = np.zeros((d, d, 3), dtype=int)

    for s_idx in range(n_stations):
        w_idx = 2 * s_idx  # wind variable index
        c_idx = 2 * s_idx + 1  # cloud variable index

        # Wind self-AR at lag 1 and 2
        G[w_idx, w_idx, 1] = 1
        G[w_idx, w_idx, 2] = 1

        # Cloud self-AR at lag 1
        G[c_idx, c_idx, 1] = 1

        # Same-station wind → cloud at lag 0 (contemporaneous)
        G[w_idx, c_idx, 0] = 1

    # Spatial wind→wind at lag 1
    for cause, effect, _ in _SPATIAL_EDGES:
        ci = 2 * STATIONS.index(cause)
        ei = 2 * STATIONS.index(effect)
        G[ci, ei, 1] = 1

    return G


def _build_weights(B1: np.ndarray, B2: np.ndarray, gamma: float) -> np.ndarray:
    """Linear weight tensor corresponding to ground_truth, shape (16, 16, 3)."""
    n_stations = len(STATIONS)
    d = 2 * n_stations
    W = np.zeros((d, d, 3))

    for s_idx in range(n_stations):
        w_idx = 2 * s_idx
        c_idx = 2 * s_idx + 1
        W[w_idx, w_idx, 1] = B1[s_idx, s_idx]
        W[w_idx, w_idx, 2] = B2[s_idx, s_idx]
        W[c_idx, c_idx, 1] = _CLOUD_RHO
        W[w_idx, c_idx, 0] = gamma

    for cause, effect, coef in _SPATIAL_EDGES:
        ci = 2 * STATIONS.index(cause)
        ei = 2 * STATIONS.index(effect)
        W[ci, ei, 1] = coef

    return W


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ercot_weather(seed: int = 42, T: int = 1460, max_lag: int = 2) -> dict:
    """Synthetic 6-hourly wind speed and cloud cover for 8 ERCOT stations.

    Generates one year (T=1460 steps x 6 h) of weather data calibrated to
    real ERCOT station observations and generation data. All values are
    stochastic -- this is NOT a copy of real data -- and reproducible via
    seed.

    Model details:

    - Wind: Weibull(k=2) with time-varying scale driven by a log-normal
      VAR(2) process. Captures diurnal/seasonal patterns,
      station-to-station lag dependencies (south-to-north prevailing
      flow), and Weibull marginals.
    - Cloud: logit-space AR(1) with contemporaneous wind coupling
      (windier = marginally lower cloud cover).
    - Solar generation: TOA irradiance (analytical) attenuated by cloud
      cover, scaled to match ERCOT solar fleet totals.
    - Wind generation: standard cubic power curve, scaled to match ERCOT
      wind fleet mean (~11.2 GW).

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    T : int
        Number of 6-hour timesteps. Default 1460 = 1 year.
    max_lag : int
        Maximum lag for ground truth graph (must be >= 2).

    Returns
    -------
    dict
        Dict with keys:

        - ``"df"`` -- DataFrame (T, 16), 6-hourly DatetimeIndex UTC
        - ``"ground_truth"`` -- ndarray (16, 16, max_lag+1), binary causal
          graph
        - ``"weights"`` -- ndarray (16, 16, max_lag+1), linear coefficients
        - ``"max_lag"`` -- int
        - ``"var_names"`` -- list of 16 column names
        - ``"stations"`` -- list of 8 station codes
        - ``"generation"`` -- dict with "wind_mwh" and "solar_mwh"
          pd.Series
    """
    if max_lag < 2:
        raise ValueError("max_lag must be >= 2 (model uses 2 wind lags).")

    rng = np.random.default_rng(seed)

    # Time index
    timestamps = pd.date_range("2023-01-01", periods=T, freq="6h", tz="UTC")

    # Station arrays
    lats = np.array([STATION_META[s]["lat"] for s in STATIONS])
    lons = np.array([STATION_META[s]["lon"] for s in STATIONS])
    cloud_means = np.array([STATION_META[s]["cloud_mean"] for s in STATIONS])

    # --- Wind ---
    B1, B2, Sigma = _build_wind_var_matrices()
    # Jensen's correction: compute exact stationary Var(r_i) via discrete Lyapunov
    # equation on the VAR(2) companion form. E[exp(r_i)] = exp(Var(r_i)/2).
    stationary_var_r = _stationary_var_lyapunov(B1, B2, Sigma)  # (8,)
    mu_log = np.column_stack(
        [
            _wind_log_scale_mean(timestamps, STATION_META[s], stationary_var_r[i])
            for i, s in enumerate(STATIONS)
        ]
    )  # (T, 8)
    W = _simulate_wind(rng, T, B1, B2, Sigma, mu_log)  # (T, 8)

    # --- Cloud ---
    C = _simulate_cloud(rng, T, cloud_means, W)  # (T, 8)

    # --- TOA + solar generation ---
    toa = _toa_irradiance(timestamps, lats, lons)  # (T, 8) W/m²
    ghi = toa * (1.0 - _CLOUD_ATTN_A * (C / 100.0) ** _CLOUD_ATTN_B)  # (T, 8)
    solar_cap_w = np.array([SOLAR_CAPACITY_WEIGHTS[s] for s in STATIONS])  # (8,)
    ghi_weighted = ghi @ solar_cap_w  # (T,) capacity-weighted GHI — zero for KELP

    # Solar efficiency noise — multiplicative factor η(t) ∈ [0.5, 1.0].
    # Represents unmodeled losses: cloud optical depth variation, panel
    # temperature derating (~0.3 %/°C above 25 °C), soiling, curtailment,
    # and maintenance windows. Efficiency cannot exceed 1 (no super-nominal
    # output), so the noise is purely downward.
    # Model: logit(η) = AR(1) around logit(0.92), so η is bounded in (0,1)
    # and typically 0.75–0.98.
    # rho=0.75 → effects persist several 6-hour periods (weather, maintenance).
    # sigma=0.20 → ~20 % logit-space noise → ~15 % CV on efficiency.
    _rho_sol, _sig_sol, _mu_eta = 0.75, 0.20, 0.92
    _mu_logit = np.log(_mu_eta / (1 - _mu_eta))  # logit(0.92) ≈ 2.44
    _z = np.zeros(T)
    _eps_sol = rng.standard_normal(T)
    for _t in range(1, T):
        _z[_t] = (
            _mu_logit + _rho_sol * (_z[_t - 1] - _mu_logit) + _sig_sol * _eps_sol[_t]
        )
    _z[0] = _mu_logit
    eta = 1.0 / (1.0 + np.exp(-_z))  # sigmoid → (0, 1), mean ≈ 0.92

    # Scale so peak solar ≈ 20,000 MW (ERCOT solar peak, summer noon ~23 GW)
    _solar_scale = 20_000.0 / max(ghi_weighted.max(), 0.01)
    solar_mwh = pd.Series(
        ghi_weighted * eta * _solar_scale, index=timestamps, name="solar_mwh"
    )

    # --- Wind generation via power curve with capacity weights ---
    wind_cap_w = np.array([WIND_CAPACITY_WEIGHTS[s] for s in STATIONS])  # (8,)
    cf = _power_curve(W)  # (T, 8) capacity factors
    cf_weighted = cf @ wind_cap_w  # (T,) weighted sum — zero weight for KELP
    # Scale so mean ≈ 11,200 MW (ERCOT wind mean, 2018-2026 average)
    _wind_scale = 11_200.0 / max(cf_weighted.mean(), 0.01)
    wind_mwh = pd.Series(cf_weighted * _wind_scale, index=timestamps, name="wind_mwh")

    # --- Assemble DataFrame ---
    var_names = []
    cols = {}
    for s_idx, s in enumerate(STATIONS):
        var_names.append(f"{s}_wind")
        var_names.append(f"{s}_cloud")
        cols[f"{s}_wind"] = W[:, s_idx]
        cols[f"{s}_cloud"] = C[:, s_idx]

    df = pd.DataFrame(cols, index=timestamps)[var_names]

    # --- Ground truth and weights ---
    ground_truth_base = _build_ground_truth()  # (16, 16, 3)
    weights_base = _build_weights(B1, B2, _WIND_CLOUD_GAMMA)

    if max_lag == 2:
        ground_truth = ground_truth_base
        weights = weights_base
    else:
        # Pad to max_lag+1 along last axis
        pad = max_lag + 1 - 3
        ground_truth = np.concatenate(
            [ground_truth_base, np.zeros((16, 16, pad), dtype=int)], axis=2
        )
        weights = np.concatenate([weights_base, np.zeros((16, 16, pad))], axis=2)

    return {
        "df": df,
        "ground_truth": ground_truth,
        "weights": weights,
        "max_lag": max_lag,
        "var_names": var_names,
        "stations": STATIONS,
        "name": "ERCOT Wind & Cloud System",
        "generation": {
            "wind_mwh": wind_mwh,
            "solar_mwh": solar_mwh,
        },
    }
