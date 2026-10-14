"""
Modelo macro-financiero de la curva de CETES (Anexo A del resumen).

Etapa 1  VAR bayesiano de Z_t = [π, brecha, fed, r28, Δln e, s91, s182, s364]
         →  E_t[Z_{t+j}] = (Σ Φ^m) A0 + Φ^j Z_t
Etapa 2  Regla de Taylor con suavizamiento:
         r28_t = ρ r28_{t-1} + (1-ρ) γᵀX_t + v_t ,  X_t = [1, π_t, brecha_t, fed_t, Δln e_{t-1}]
Etapa 3  Componente de expectativas y prima por plazo para M = 3, 6, 12:
         Ā_t^(M) = (1/M)[ r28_t (1-ρ^M)/(1-ρ) + Σ_{j=1}^{M-1} (1-ρ^{M-j}) γᵀE_t[X_{t+j}] ]
         θ_{M,t} = s_{M,t} − (Ā_t^(M) − r28_t)
Etapa 4  Contraste con la expectativa implícita del VAR: Ā^VAR = (1/M) Σ_{k<M} E_t[r28_{t+k}]
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from statsmodels.tsa.filters.hp_filter import hpfilter

from models.Cholesky_esferico import cholesky_esferico

VARIABLES_Z = ["pi", "brecha", "fed", "r28", "dlne", "s91", "s182", "s364"]
PLAZOS_M = {"s91": 3, "s182": 6, "s364": 12}


# ======================================================================
# Construcción de Z_t
# ======================================================================
def construir_Z(piv, macro, lamb_hp=129600):
    """
    piv:   tabla de tasas de CETES (columnas 28, 91, 182, 364), índice mensual.
    macro: salida de Data.macro.macro_call().
    """
    curva = piv[[28, 91, 182, 364]].copy()
    curva.index = pd.to_datetime(curva.index).to_period("M").to_timestamp()

    m = macro.copy()
    pi = 100 * (np.log(m["inpc"]) - np.log(m["inpc"].shift(12)))           # inflación anual, %
    act = np.log(m["actividad"]).dropna()
    _, tendencia = hpfilter(act, lamb=lamb_hp)
    brecha = 100 * (act - tendencia)                                        # brecha del producto, %
    dlne = 100 * np.log(m["tc"]).diff()                                     # depreciación mensual, %

    Z = pd.DataFrame({
        "pi": pi,
        "brecha": brecha,
        "fed": m["fedfunds"],
        "r28": curva[28],
        "dlne": dlne,
        "s91": curva[91] - curva[28],
        "s182": curva[182] - curva[28],
        "s364": curva[364] - curva[28],
    })[VARIABLES_Z].dropna()

    print(f"\nZ_t: {Z.index.min():%Y-%m} → {Z.index.max():%Y-%m}  ({len(Z)} meses, {Z.shape[1]} variables)")
    return Z


# ======================================================================
# Etapa 1: VAR bayesiano general (Φ constante, R con Cholesky esférico)
# ======================================================================
class BayesianVAR:

    def __init__(self, Z):
        self.Z = Z
        self.vars = list(Z.columns)
        self.lags = [f"{v}(t-1)" for v in self.vars]
        d = pd.concat([Z, Z.shift(1).set_axis(self.lags, axis=1)], axis=1).dropna()
        self.fechas = d.index
        self.y_raw = d[self.vars].values
        self.x_raw = d[self.lags].values
        self.mu = Z.mean().values
        self.sd = Z.std(ddof=0).values
        self.idata = None

    def fit(self, vol_estocastica=False, eta=2.0, media_propia=0.8, sd_propia=0.3, sd_cruzada=0.2,
            draws=1000, tune=1000, chains=4, target_accept=0.9, seed=42, nuts_sampler="nutpie"):
        """
        Prior tipo Minnesota en escala estandarizada:
            Φ_ii ~ N(media_propia, sd_propia²)   (persistencia propia)
            Φ_ij ~ N(0, sd_cruzada²)             (efectos cruzados, encogidos hacia 0)
        """
        y = (self.y_raw - self.mu) / self.sd
        X = (self.x_raw - self.mu) / self.sd
        T, k = y.shape
        coords = {"eq": self.vars, "eq_": self.vars, "reg": self.lags}

        media = np.where(np.eye(k, dtype=bool), media_propia, 0.0)
        escala = np.where(np.eye(k, dtype=bool), sd_propia, sd_cruzada)

        with pm.Model(coords=coords) as self.model:
            a_s   = pm.Normal("a_s", 0, 1, dims="eq")
            Phi_s = pm.Normal("Phi_s", media, escala, dims=("eq", "reg"))
            mu = a_s + pt.dot(X, Phi_s.T)                                     # (T, k)

            L_R = cholesky_esferico("R", N=k, eta=eta)

            if vol_estocastica:
                omega = pm.HalfNormal("omega", 0.1, dims="eq")
                h0    = pm.Normal("h0", -1, 1, dims="eq")
                z_h   = pm.Normal("z_h", 0, 1, shape=(T, k))
                sig_s = pt.exp((h0 + omega * pt.cumsum(z_h, axis=0)) / 2)    # (T, k)
                chol = sig_s[:, :, None] * L_R[None, :, :]
                pm.Deterministic("sigma", sig_s * self.sd)
            else:
                sig_s = pm.HalfNormal("sigma_s", 1, dims="eq")                # (k,)
                chol = sig_s[:, None] * L_R                                   # un solo Cholesky: rápido
                pm.Deterministic("sigma", sig_s * self.sd, dims="eq")

            pm.MvNormal("obs", mu=mu, chol=chol, observed=y)

            # --- unidades originales: Z_t = A0 + Φ Z_{t-1} + ε_t ---
            Phi = pm.Deterministic("Phi", Phi_s * (self.sd[:, None] / self.sd[None, :]),
                                   dims=("eq", "reg"))
            pm.Deterministic("A0", self.mu + self.sd * a_s - pt.dot(Phi, self.mu), dims="eq")

            self.idata = pm.sample(draws=draws, tune=tune, chains=chains,
                                   target_accept=target_accept, random_seed=seed,
                                   nuts_sampler=nuts_sampler)
            pm.compute_log_likelihood(self.idata)
        return self

    def save(self, path):
        self.idata.to_netcdf(path)

    def load(self, path):
        self.idata = az.from_netcdf(path)
        return self

    def diagnostics(self):
        print(az.summary(self.idata, var_names=["Phi", "R_theta"]))
        print("Divergencias:", int(self.idata.sample_stats["diverging"].sum()))
        Phi = self.tabla("Phi")
        radio = np.abs(np.linalg.eigvals(Phi.values)).max()
        print(f"Radio espectral de Φ (media posterior): {radio:.4f}  "
              f"({'estable' if radio < 1 else 'NO estable'})")

    def tabla(self, nombre):
        M = self.idata.posterior[nombre].mean(("chain", "draw")).values
        cols = self.lags if nombre == "Phi" else self.vars
        return pd.DataFrame(M, index=self.vars, columns=cols)

    def muestras(self, n, rng):
        post = self.idata.posterior
        A0 = post["A0"].stack(s=("chain", "draw")).values.T                       # (S, k)
        Phi = post["Phi"].stack(s=("chain", "draw")).transpose("s", ...).values   # (S, k, k)
        idx = rng.choice(len(A0), size=n, replace=False)
        return A0[idx], Phi[idx]


# ======================================================================
# Etapa 2: regla de Taylor bayesiana con suavizamiento
# ======================================================================
class TaylorRuleBayes:
    """
    r28_t = ρ r28_{t-1} + (1-ρ)(γ0 + γ_π π_t + γ_y brecha_t + γ_F fed_t + γ_e Δln e_{t-1}) + v_t
    (Δln e entra rezagado para evitar simultaneidad con las sorpresas de Banxico.)
    """
    NOMBRES = ["g0", "g_pi", "g_y", "g_fed", "g_e"]

    def __init__(self, Z):
        d = pd.DataFrame({
            "r28": Z["r28"], "r28_lag": Z["r28"].shift(1),
            "pi": Z["pi"], "brecha": Z["brecha"], "fed": Z["fed"], "dlne_lag": Z["dlne"].shift(1),
        }).dropna()
        self.d = d
        self.fechas = d.index
        self.idata = None

    def fit(self, draws=1000, tune=1000, chains=4, target_accept=0.95, seed=42, nuts_sampler="nutpie"):
        d = self.d
        with pm.Model() as self.model:
            rho   = pm.Beta("rho", 8, 2)                  # suavizamiento: centrado en ~0.8
            g0    = pm.Normal("g0", 0, 5)
            g_pi  = pm.Normal("g_pi", 1.5, 0.5)           # principio de Taylor: > 1
            g_y   = pm.Normal("g_y", 0.5, 0.5)
            g_fed = pm.Normal("g_fed", 0.5, 0.5)
            g_e   = pm.Normal("g_e", 0, 1)
            sigma_v = pm.HalfNormal("sigma_v", 1)

            i_star = pm.Deterministic(
                "i_star",
                g0 + g_pi * d["pi"].values + g_y * d["brecha"].values
                + g_fed * d["fed"].values + g_e * d["dlne_lag"].values,
            )
            mu = rho * d["r28_lag"].values + (1 - rho) * i_star
            pm.Normal("obs", mu=mu, sigma=sigma_v, observed=d["r28"].values)

            self.idata = pm.sample(draws=draws, tune=tune, chains=chains,
                                   target_accept=target_accept, random_seed=seed,
                                   nuts_sampler=nuts_sampler)
            pm.compute_log_likelihood(self.idata)
        return self

    def save(self, path):
        self.idata.to_netcdf(path)

    def load(self, path):
        self.idata = az.from_netcdf(path)
        return self

    def diagnostics(self):
        print(az.summary(self.idata, var_names=["rho"] + self.NOMBRES + ["sigma_v"]))
        print("Divergencias:", int(self.idata.sample_stats["diverging"].sum()))
        rho = self.idata.posterior["rho"].values.ravel()
        print(f"Vida media de una brecha (i* − i): {np.median(np.log(0.5) / np.log(rho)):.1f} meses")

    def muestras(self, n, rng):
        post = self.idata.posterior
        rho = post["rho"].values.ravel()
        gam = np.stack([post[g].values.ravel() for g in self.NOMBRES], axis=1)   # (S, 5)
        idx = rng.choice(len(rho), size=n, replace=False)
        return rho[idx], gam[idx]

    def plot(self):
        post = self.idata.posterior
        i_star = post["i_star"].median(("chain", "draw")).values
        plt.figure(figsize=(12, 4))
        plt.plot(self.fechas, self.d["r28"], label="r28 observada")
        plt.plot(self.fechas, i_star, label="i* (tasa deseada según la regla)", ls="--")
        plt.title("Regla de Taylor: tasa observada vs tasa deseada")
        plt.ylabel("%"); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()


# ======================================================================
# Etapas 3 y 4: componente de expectativas y prima por plazo
# ======================================================================
class TermPremium:

    def __init__(self, var, regla, n_muestras=400, seed=0):
        self.var, self.regla = var, regla
        self.n = n_muestras
        self.rng = np.random.default_rng(seed)
        self.resultados = None

    def calcular(self):
        Z = self.var.Z
        fechas = Z.index.intersection(self.regla.fechas)
        Zv = Z.loc[fechas].values                                   # (T, k)
        col = {v: i for i, v in enumerate(self.var.vars)}
        Mmax = max(PLAZOS_M.values())

        A0, Phi = self.var.muestras(self.n, self.rng)               # (n, k), (n, k, k)
        rho, gam = self.regla.muestras(self.n, self.rng)            # (n,), (n, 5)

        # E_t[Z_{t+j}], j = 0..Mmax-1, para cada muestra: (Mmax, n, T, k)
        E = np.empty((Mmax, self.n, len(fechas), Zv.shape[1]))
        E[0] = Zv[None]
        for j in range(1, Mmax):
            E[j] = A0[:, None, :] + np.einsum("nik,ntk->nti", Phi, E[j - 1])

        # γᵀ E_t[X_{t+j}]   con X = [1, π, brecha, fed, Δln e (rezagado)]
        def g(j):
            return (gam[:, [0]]
                    + gam[:, [1]] * E[j][..., col["pi"]]
                    + gam[:, [2]] * E[j][..., col["brecha"]]
                    + gam[:, [3]] * E[j][..., col["fed"]]
                    + gam[:, [4]] * E[j - 1][..., col["dlne"]])     # (n, T)

        r28 = Zv[:, col["r28"]][None]                               # (1, T)
        rho_ = rho[:, None]
        res = {}
        for s, M in PLAZOS_M.items():
            suma = sum((1 - rho_ ** (M - j)) * g(j) for j in range(1, M))
            A_regla = (r28 * (1 - rho_ ** M) / (1 - rho_) + suma) / M
            A_var = E[:M, :, :, col["r28"]].mean(axis=0)
            spread = Zv[:, col[s]][None]

            res[s] = {
                "spread": Zv[:, col[s]],
                "exp_regla": A_regla - r28,
                "exp_var": A_var - r28,
                "theta_regla": spread - (A_regla - r28),
                "theta_var": spread - (A_var - r28),
            }
        self.fechas = fechas
        self.resultados = res
        return self

    def series(self, plazo="s364", prob=0.90):
        """DataFrame con mediana y bandas de cada componente para un plazo."""
        r = self.resultados[plazo]
        a = (1 - prob) / 2
        out = {"spread": r["spread"]}
        for k in ["exp_regla", "exp_var", "theta_regla", "theta_var"]:
            out[k] = np.median(r[k], axis=0)
            out[f"{k}_lo"] = np.quantile(r[k], a, axis=0)
            out[f"{k}_hi"] = np.quantile(r[k], 1 - a, axis=0)
        return pd.DataFrame(out, index=self.fechas)

    def resumen(self):
        filas = []
        for s, M in PLAZOS_M.items():
            df = self.series(s)
            filas.append({
                "plazo": f"{s} (M={M})",
                "spread medio": df["spread"].mean(),
                "expectativa (regla)": df["exp_regla"].mean(),
                "prima θ (regla)": df["theta_regla"].mean(),
                "expectativa (VAR)": df["exp_var"].mean(),
                "prima θ (VAR)": df["theta_var"].mean(),
                "corr(θ_regla, θ_VAR)": df["theta_regla"].corr(df["theta_var"]),
            })
        return pd.DataFrame(filas).set_index("plazo")

    def plot(self, prob=0.90):
        fig, ax = plt.subplots(len(PLAZOS_M), 1, figsize=(13, 11), sharex=True)
        for a, (s, M) in zip(ax, PLAZOS_M.items()):
            df = self.series(s, prob)
            f = df.index
            a.plot(f, df["spread"], color="black", lw=1.2, label="spread observado")
            a.plot(f, df["exp_regla"], label="expectativa de Banxico (regla)")
            a.fill_between(f, df["exp_regla_lo"], df["exp_regla_hi"], alpha=0.2)
            a.plot(f, df["theta_regla"], label="prima por plazo θ (regla)")
            a.fill_between(f, df["theta_regla_lo"], df["theta_regla_hi"], alpha=0.2)
            a.plot(f, df["theta_var"], ls=":", label="θ (expectativa del VAR)")
            a.axhline(0, color="gray", lw=0.8)
            a.set(title=f"{s}: spread = expectativa + prima   (M = {M} meses)", ylabel="puntos %")
            a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper left")
        plt.tight_layout(); plt.show()
