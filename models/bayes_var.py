"""
TVP-VAR bayesiano con volatilidad estocástica y correlación entre choques.

    Y_t = B0 + B_t · Y_{t-1} + ε_t ,          ε_t ~ N(0, D_t R D_t)

    Y_t      = [r28, s91, s182, s364]_t        (s_i = r_i − r28)
    B_t      = B̄ + caminata aleatoria centrada (una τ por coeficiente, Q diagonal)
    D_t      = diag(σ_1t, …, σ_kt),  log σ²_it = caminata aleatoria (volatilidad estocástica)
    R        = matriz de correlación, Cholesky esférico + prior LKJ(η)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from models.Cholesky_esferico import cholesky_esferico


class BayesianTVPVAR:

    def __init__(self, piv, base=28, plazos_spread=(91, 182, 364)):
        Y = pd.DataFrame({f"r{base}": piv[base]})
        for m in plazos_spread:
            Y[f"s{m}"] = piv[m] - piv[base]
        Y = Y.dropna()
        self.vars = list(Y.columns)
        self.lags = [f"{v}(t-1)" for v in self.vars]

        d = pd.concat([Y, Y.shift(1).set_axis(self.lags, axis=1)], axis=1).dropna()
        self.fechas = d.index
        self.y_raw = d[self.vars].values           # (T, k)
        self.x_raw = d[self.lags].values           # (T, k)

        # misma estandarización para Y y sus rezagos (son las mismas variables)
        self.mu = Y.mean().values
        self.sd = Y.std(ddof=0).values
        self.idata = None

    @classmethod
    def from_regressors(cls, rg, **kwargs):
        return cls(rg.piv, **kwargs)

    # ------------------------------------------------------------------
    def fit(self, tvp=True, vol_estocastica=True, correlacion=True,
            tau_scale=0.02, eta=2.0,
            draws=1000, tune=2000, chains=4, target_accept=0.99, seed=42,
            nuts_sampler="nutpie"):
        y = (self.y_raw - self.mu) / self.sd
        X = (self.x_raw - self.mu) / self.sd
        T, k = y.shape
        coords = {"eq": self.vars, "eq_": self.vars, "reg": self.lags}

        with pm.Model(coords=coords) as self.model:
            a_s    = pm.Normal("a_s", 0, 1, dims="eq")
            Bbar_s = pm.Normal("Bbar_s", 0, 1, dims=("eq", "reg"))

            # ---- B_t: caminata aleatoria centrada, Q diagonal ----
            if tvp:
                tau = pm.HalfNormal("tau", tau_scale, dims=("eq", "reg"))
                z_B = pm.Normal("z_B", 0, 1, shape=(T, k, k))
                Sz  = pt.cumsum(z_B, axis=0)
                B_s = Bbar_s + tau * (Sz - Sz.mean(axis=0))            # (T, k, k)
                mu  = a_s + (B_s * X[:, None, :]).sum(axis=2)          # (T, k)
            else:
                B_s = Bbar_s
                mu  = a_s + pt.dot(X, Bbar_s.T)

            # ---- D_t: volatilidad estocástica por ecuación ----
            if vol_estocastica:
                omega = pm.HalfNormal("omega", 0.1, dims="eq")
                h0    = pm.Normal("h0", -1, 1, dims="eq")
                z_h   = pm.Normal("z_h", 0, 1, shape=(T, k))
                sig_s = pt.exp((h0 + omega * pt.cumsum(z_h, axis=0)) / 2)   # (T, k)
            else:
                sig_s = pm.HalfNormal("sigma_s", 1, dims="eq") * pt.ones((T, k))

            # ---- R: Cholesky esférico con prior LKJ(η) ----
            if correlacion:
                L_R = cholesky_esferico("R", N=k, eta=eta)             # crea R, R_L, R_theta
            else:
                L_R = pt.eye(k)

            # Cholesky de Σ_t = D_t R D_t  →  diag(σ_t) · L_R
            chol_t = sig_s[:, :, None] * L_R[None, :, :]               # (T, k, k)
            pm.MvNormal("obs", mu=mu, chol=chol_t, observed=y)

            # ---- unidades originales ----
            escala = self.sd[:, None] / self.sd[None, :]               # sd_i / sd_j
            pm.Deterministic("Bbar", Bbar_s * escala, dims=("eq", "reg"))
            pm.Deterministic("sigma", sig_s * self.sd)                 # (T, k)
            if tvp:
                pm.Deterministic("B_t", B_s * escala)                  # (T, k, k)
                pm.Deterministic("tau_orig", tau * escala, dims=("eq", "reg"))

            self.idata = pm.sample(
                draws=draws, tune=tune, chains=chains,
                target_accept=target_accept, random_seed=seed,
                nuts_sampler=nuts_sampler,
            )
            pm.compute_log_likelihood(self.idata)
        return self

    # ------------------------------------------------------------------
    def save(self, path):
        self.idata.to_netcdf(path)

    def load(self, path):
        self.idata = az.from_netcdf(path)
        return self

    def _check(self):
        if self.idata is None:
            raise RuntimeError("Primero ejecuta .fit() o .load()")

    def diagnostics(self):
        self._check()
        vars_ = [v for v in ["Bbar", "tau_orig", "omega", "R_theta"]
                 if v in self.idata.posterior]
        print(az.summary(self.idata, var_names=vars_))
        print("Divergencias:", int(self.idata.sample_stats["diverging"].sum()))

    # ------------------------------------------------------------------
    def tabla(self, nombre):
        """Media posterior de una matriz k×k ('Bbar' o 'R') como DataFrame."""
        self._check()
        M = self.idata.posterior[nombre].mean(("chain", "draw")).values
        cols = self.lags if nombre == "Bbar" else self.vars
        return pd.DataFrame(M, index=self.vars, columns=cols)

    def tabla_R_intervalos(self, prob=0.94):
        """Correlaciones con intervalo de credibilidad."""
        self._check()
        R = self.idata.posterior["R"].stack(s=("chain", "draw")).values   # (k, k, S)
        a = (1 - prob) / 2
        filas = []
        for i in range(len(self.vars)):
            for j in range(i):
                r = R[i, j]
                filas.append({
                    "par": f"{self.vars[i]} – {self.vars[j]}",
                    "media": r.mean(),
                    "lo": np.quantile(r, a),
                    "hi": np.quantile(r, 1 - a),
                    "P(r>0)": (r > 0).mean(),
                })
        return pd.DataFrame(filas).set_index("par")

    def plot_B_t(self, prob=0.94):
        self._check()
        Bt = self.idata.posterior["B_t"]
        a = (1 - prob) / 2
        med = Bt.mean(("chain", "draw")).values
        lo  = Bt.quantile(a, dim=("chain", "draw")).values
        hi  = Bt.quantile(1 - a, dim=("chain", "draw")).values
        k, f = len(self.vars), self.fechas

        fig, ax = plt.subplots(k, k, figsize=(16, 12), sharex=True)
        for i in range(k):
            for j in range(k):
                ax[i, j].plot(f, med[:, i, j], lw=1.5)
                ax[i, j].fill_between(f, lo[:, i, j], hi[:, i, j], alpha=0.3)
                ax[i, j].axhline(0, color="gray", lw=0.8)
                ax[i, j].set_title(f"B{i+1}{j+1}: {self.lags[j]} → {self.vars[i]}(t)", fontsize=9)
                ax[i, j].grid(alpha=0.3)
        fig.suptitle("Bₜ en el tiempo", fontsize=13)
        plt.tight_layout(); plt.show()

        radio = np.abs(np.linalg.eigvals(med)).max(axis=1)
        plt.figure(figsize=(12, 3.5))
        plt.plot(f, radio); plt.axhline(1, color="red", ls="--", label="|λ| = 1")
        plt.title("Radio espectral de Bₜ"); plt.grid(alpha=0.3); plt.legend()
        plt.tight_layout(); plt.show()

    def plot_sigma(self):
        self._check()
        s = self.idata.posterior["sigma"].mean(("chain", "draw")).values   # (T, k)
        plt.figure(figsize=(12, 4))
        for i, v in enumerate(self.vars):
            plt.plot(self.fechas, s[:, i], label=v)
        plt.title("Dₜ: volatilidad de cada ecuación"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.show()
