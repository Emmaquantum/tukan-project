import numpy as np
import pymc as pm
import pytensor.tensor as pt


# ======================================================================
# Versión NumPy (para verificar y para inicializar)
# ======================================================================
def angulos_a_cholesky_np(theta, N):
    """θ (M,) → L (N, N)."""
    L = np.zeros((N, N))
    L[0, 0] = 1.0
    idx = 0
    for i in range(1, N):
        prod_sin = 1.0
        for j in range(i):
            L[i, j] = prod_sin * np.cos(theta[idx + j])
            prod_sin *= np.sin(theta[idx + j])
        L[i, i] = prod_sin
        idx += i
    return L


def cholesky_a_angulos_np(L):
    """Inversa: L (N, N) → θ (M,). Útil para dar valores iniciales."""
    N = L.shape[0]
    theta = []
    for i in range(1, N):
        prod_sin = 1.0
        for j in range(i):
            c = np.clip(L[i, j] / prod_sin, -1.0, 1.0)
            th = np.arccos(c)
            theta.append(th)
            prod_sin *= np.sin(th)
    return np.array(theta)


def exponentes_lkj(N, eta):
    """Exponente de sin(θ_ik) en el prior LKJ, en el mismo orden que θ."""
    return np.array([N - (k + 1) + 2 * eta - 2
                     for i in range(1, N) for k in range(i)], dtype=float)


# ======================================================================
# Versión PyTensor (para usar dentro de un pm.Model)
# ======================================================================
def cholesky_esferico(nombre, N, eta=2.0):
    """
    Crea dentro del modelo activo:
        {nombre}_y      ∈ ℝ^M        variable libre (lo que muestrea NUTS)
        {nombre}_theta  ∈ (0, π)^M   ángulos
        {nombre}_L      Cholesky de la correlación
        {nombre}        R = L Lᵀ
    y devuelve L (tensor N×N).
    """
    M = N * (N - 1) // 2

    # --- Paso 1: vector sin restricciones ---
    y = pm.Flat(f"{nombre}_y", shape=M)

    # --- Paso 2: y → θ ∈ (0, π) con la logística ---
    theta = pm.Deterministic(f"{nombre}_theta", np.pi * pm.math.sigmoid(y))

    # log |dθ/dy| = log π + log s + log(1 - s),  con s = sigmoid(y)
    #   log s = -softplus(-y),  log(1 - s) = -softplus(y)   (numéricamente estable)
    log_jac = pt.sum(np.log(np.pi) - pt.softplus(-y) - pt.softplus(y))

    # Prior LKJ(η) sobre los ángulos
    exps = exponentes_lkj(N, eta)
    log_prior = pt.sum(exps * pt.log(pt.sin(theta)))

    pm.Potential(f"{nombre}_lkj", log_prior + log_jac)

    # --- Paso 3: construir L fila por fila ---
    filas = []
    idx = 0
    for i in range(N):
        fila = []
        prod_sin = pt.constant(1.0)
        for j in range(N):
            if j < i:
                th = theta[idx + j]
                fila.append(prod_sin * pt.cos(th))
                prod_sin = prod_sin * pt.sin(th)
            elif j == i:
                fila.append(prod_sin)                 # L_ii = Π sin θ_ik  (1 en la fila 0)
            else:
                fila.append(pt.constant(0.0))
        idx += i
        filas.append(pt.stack(fila))
    L = pm.Deterministic(f"{nombre}_L", pt.stack(filas))

    # --- Paso 4: R = L Lᵀ ---
    pm.Deterministic(nombre, L @ L.T)
    return L