import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
import arviz as az


class BayesianTVP:
    def __init__(self, y, x, y_name="y", x_name="spread", ols=None):
        self.d = pd.concat([y.rename(y_name), x.rename(x_name)], axis=1).dropna()
        self.y_name, self.x_name = y_name, x_name
        self.fechas = self.d.index
        self.ols = ols
        self.idata = None

        # estadísticos de estandarización (disponibles aunque se cargue desde archivo)
        self.y_mu = self.d[y_name].mean()
        self.y_sd = self.d[y_name].std(ddof=0)
        self.x_sd = self.d[x_name].std(ddof=0)

    @classmethod
    def from_regressors(cls, rg):
        return cls(rg.mov, rg.spread, ols=rg.m2)

    # ------------------------------------------------------------------
    def _priors_ols(self, inflacion):
        """Traduce α, β de OLS a la escala estandarizada."""
        if self.ols is None:
            return dict(a_mu=0, a_sd=1, b_mu=0, b_sd=1, s_scale=1)

        a, b = self.ols.params["const"], self.ols.params[self.x_name]
        se_a, se_b = self.ols.bse["const"], self.ols.bse[self.x_name]
        resid_sd = self.ols.resid.std()

        return dict(
            a_mu=(a - self.y_mu) / self.y_sd,
            a_sd=inflacion * se_a / self.y_sd,
            b_mu=b * self.x_sd / self.y_sd,
            b_sd=inflacion * se_b * self.x_sd / self.y_sd,
            s_scale=2 * resid_sd / self.y_sd,
        )

    # ------------------------------------------------------------------
    def fit(self, draws=2000, tune=3000, chains=4, target_accept=0.99, seed=42,
        vol_estocastica=False, inflacion=3.0):
        # estandarización con los estadísticos calculados en __init__
        y = (self.d[self.y_name].values - self.y_mu) / self.y_sd
        x = self.d[self.x_name].values / self.x_sd
        T = len(y)

        p = self._priors_ols(inflacion)
        self.priors = p

        with pm.Model() as self.model:
            # --- priors informados por OLS ---
            alpha    = pm.Normal("alpha_s", p["a_mu"], p["a_sd"])
            beta_bar = pm.Normal("beta_bar_s", p["b_mu"], p["b_sd"])

            # --- β_t = promedio + desviación (caminata centrada) ---
            tau    = pm.HalfNormal("tau", 0.03)
            z_beta = pm.Normal("z_beta", 0, 1, shape=T)
            S      = pt.cumsum(z_beta)
            beta_s = beta_bar + tau * (S - S.mean())

            # --- volatilidad ---
            if vol_estocastica:
                omega   = pm.HalfNormal("omega", 0.05)
                h0      = pm.Normal("h0", np.log(p["s_scale"] ** 2 / 4), 1)
                z_h     = pm.Normal("z_h", 0, 1, shape=T)
                sigma_s = pm.math.exp((h0 + omega * pt.cumsum(z_h)) / 2)
            else:
                sigma_s = pm.HalfNormal("sigma_s", p["s_scale"])

            pm.Normal("obs", mu=alpha + beta_s * x, sigma=sigma_s, observed=y)

            # --- de regreso a unidades originales ---
            pm.Deterministic("alpha",    alpha * self.y_sd + self.y_mu)
            pm.Deterministic("beta_bar", beta_bar * self.y_sd / self.x_sd)
            pm.Deterministic("beta",     beta_s * self.y_sd / self.x_sd)
            pm.Deterministic("sigma",    sigma_s * self.y_sd * pt.ones(T))
            pm.Deterministic("tau_orig", tau * self.y_sd / self.x_sd)

            self.idata = pm.sample(
                draws=draws, tune=tune, chains=chains,
                target_accept=target_accept, random_seed=seed,
                nuts_sampler="nutpie"
            )
            pm.compute_log_likelihood(self.idata) 
        return self

    # ------------------------------------------------------------------
    def diagnostics(self):
        self._check()
        vars_ = ["alpha", "beta_bar", "tau"]
        if "omega" in self.idata.posterior:
            vars_.append("omega")
        print(az.summary(self.idata, var_names=vars_))
        print("Divergencias:", int(self.idata.sample_stats["diverging"].sum()))

        if self.ols is not None:
            post = self.idata.posterior
            print("\nComparación con OLS:")
            print(f"  α:  OLS = {self.ols.params['const']:.3f}   "
                  f"Bayes = {float(post['alpha'].mean()):.3f}")
            print(f"  β̄:  OLS = {self.ols.params[self.x_name]:.3f}   "
                  f"Bayes = {float(post['beta_bar'].mean()):.3f}")

    # ------------------------------------------------------------------
    def save(self, path="idata_tvp.nc"):
        """Guarda la posterior para no volver a muestrear."""
        self.idata.to_netcdf(path)

    def load(self, path="idata_tvp.nc"):
        self.idata = az.from_netcdf(path)
        return self

    # ------------------------------------------------------------------
    def _check(self):
        if self.idata is None:
            raise RuntimeError("Primero ejecuta .fit() o .load()")

    def results(self, hdi_prob=0.94):
        """DataFrame con β_t, su HDI, P(β_t > 1) y σ_t por fecha."""
        self._check()
        post = self.idata.posterior
        b_hdi = az.hdi(self.idata, var_names=["beta"], hdi_prob=hdi_prob)["beta"].values
        s_hdi = az.hdi(self.idata, var_names=["sigma"], hdi_prob=hdi_prob)["sigma"].values

        return pd.DataFrame({
            "beta_med":  post["beta"].median(("chain", "draw")).values,
            "beta_lo":   b_hdi[:, 0],
            "beta_hi":   b_hdi[:, 1],
            "p_beta_1":  (post["beta"] > 1).mean(("chain", "draw")).values,
            "sigma_med": post["sigma"].median(("chain", "draw")).values,
            "sigma_lo":  s_hdi[:, 0],
            "sigma_hi":  s_hdi[:, 1],
        }, index=self.fechas)

    # ------------------------------------------------------------------
    def plot(self, hdi_prob=0.94):
        r = self.results(hdi_prob)
        f = r.index
        pct = int(hdi_prob * 100)

        fig, ax = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

        ax[0].plot(f, r.beta_med, label="β mediana posterior")
        ax[0].fill_between(f, r.beta_lo, r.beta_hi, alpha=0.3, label=f"HDI {pct}%")
        ax[0].axhline(1, color="red", ls="--", label="β = 1 (hipótesis de expectativas)")
        ax[0].axhline(0, color="gray", lw=0.8)
        ax[0].set(title="βₜ: sensibilidad de la relación en el tiempo", ylabel="β")
        ax[0].legend(); ax[0].grid(alpha=0.3)

        ax[1].plot(f, r.p_beta_1, color="darkorange")
        ax[1].axhline(0.5, color="gray", ls=":")
        ax[1].set(title="P(βₜ > 1 | datos)", ylabel="probabilidad", ylim=(0, 1))
        ax[1].grid(alpha=0.3)

        ax[2].plot(f, r.sigma_med, color="purple", label="σ mediana posterior")
        ax[2].fill_between(f, r.sigma_lo, r.sigma_hi, alpha=0.3, color="purple")
        ax[2].set(title="σₜ: volatilidad del error", ylabel="σ")
        ax[2].legend(); ax[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()