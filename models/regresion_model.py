import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

import logging
logging.getLogger("pytensor.graph.rewriting.basic").setLevel(logging.CRITICAL)

class Regressors:
    def __init__(self, df):
        self.df = df.copy()
        self._preparar()

    def _preparar(self):
        d = self.df.copy()
        d["DIAS"] = d["Contract-time"].str.extract(r"(\d+)")[0].astype(int)
        self.piv = d.pivot(index="DATE", columns="DIAS", values="VALUE_PCT").sort_index()
        r28, r364 = self.piv[28], self.piv[364]

        self.spread = (r364 - r28).rename("spread")
        self.eps    = self.spread.abs().rename("eps")
        self.mov    = (r28.shift(-12) - r28).rename("y")
        self.y_eps  = self.mov.abs().rename("y_eps")

    def ajustar(self, y, x):
        d = pd.concat([y, x], axis=1).dropna()
        X = sm.add_constant(d[x.name])
        m = sm.OLS(d[y.name], X).fit(cov_type="HAC", cov_kwds={"maxlags": 11})
        return m, d

    def fit_model(self):
        self.m1, self.d1 = self.ajustar(self.y_eps, self.eps)
        self.m2, self.d2 = self.ajustar(self.mov, self.spread)
        return self

    def comparative_table(self):
        m1, m2 = self.m1, self.m2
        # ---------- Tabla comparativa ----------
        resumen = pd.DataFrame({
            "Modelo 1 (|·|)": {
                "pendiente":   m1.params["eps"],
                "intercepto":  m1.params["const"],
                "error est.":  m1.bse["eps"],
                "p-valor":     m1.pvalues["eps"],
                "R²":          m1.rsquared,
                "n":           int(m1.nobs),
            },
            "Modelo 2 (signo)": {
                "pendiente":   m2.params["spread"],
                "intercepto":  m2.params["const"],
                "error est.":  m2.bse["spread"],
                "p-valor":     m2.pvalues["spread"],
                "R²":          m2.rsquared,
                "n":           int(m2.nobs),
            },
        })
        print(resumen.round(3))

        print("---------SUMMARY----------")
        print("-----------------------------------------RESUMEN DEL MODELO 1 - SUMMARY-----------------------------------------")
        print(m1.summary())
        print("-----------------------------------------RESUMEN DEL MODELO 2 - SUMMARY-----------------------------------------")
        print(m2.summary())

    def graficar_model(self):
        # ---------- Gráficas ----------
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))

        for a, d, m, x, y, titulo in [
            (ax[0], self.d1, self.m1, "eps",    "y_eps", "Modelo 1: y_ε = β₁·ε + β₀"),
            (ax[1], self.d2, self.m2, "spread", "y",     "Modelo 2: y = β₂·(X³⁶⁴ − X²⁸) + β₃"),
        ]:
            a.scatter(d[x], d[y], alpha=0.5, s=18)
            xs = d[x].sort_values()
            a.plot(xs, m.params["const"] + m.params[x] * xs, color="red", linewidth=2,
                label=f"β = {m.params[x]:.2f}   R² = {m.rsquared:.2f}")
            a.axhline(0, color="gray", linewidth=0.8)
            a.axvline(0, color="gray", linewidth=0.8)
            a.set(title=titulo, xlabel=x, ylabel=y)
            a.grid(alpha=0.3)
            a.legend()

        plt.tight_layout()
        plt.show()