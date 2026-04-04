import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


def generar_caso_de_uso_diagnosticar_dataset():
    """
    Genera un par (input, output) aleatorio para la función diagnosticar_dataset.

    Returns:
        tuple: (input_dict, output_esperado)
            - input_dict: dict con claves 'df', 'target_col', 'umbral_varianza',
              'umbral_correlacion'
            - output_esperado: dict con claves 'porcentaje_nulos',
              'features_cuasi_constantes', 'pares_redundantes',
              'n_features_original', 'n_features_utiles'
    """
    rng = np.random.default_rng()

    # --- Parámetros aleatorios ---
    n_filas = int(rng.integers(50, 151))        # 50 a 150 filas
    n_features = int(rng.integers(5, 11))       # 5 a 10 features
    umbral_varianza = round(float(rng.choice([0.005, 0.01, 0.02, 0.05])), 3)
    umbral_correlacion = round(float(rng.choice([0.85, 0.9, 0.92, 0.95])), 2)

    # Nombres de columnas aleatorios
    pool_nombres = [
        "presion", "humedad", "radiacion", "viento", "caudal",
        "nivel_rio", "temperatura", "salinidad", "turbidez",
        "oxigeno", "conductividad", "ph", "altitud", "densidad"
    ]
    nombres_features = sorted(rng.choice(pool_nombres, size=n_features, replace=False).tolist())

    pool_targets = ["calidad_agua", "indice_riesgo", "categoria_alerta", "nivel_critico"]
    target_col = str(rng.choice(pool_targets))

    # --- Construir DataFrame con características realistas ---
    data = {}

    # Features normales con distintas escalas
    for nombre in nombres_features:
        media = float(rng.uniform(-50, 200))
        std = float(rng.uniform(1, 50))
        data[nombre] = rng.normal(media, std, size=n_filas)

    # Inyectar 1-2 features cuasi-constantes (reemplazar columnas existentes)
    n_cuasi = int(rng.integers(1, 3))
    indices_cuasi = rng.choice(len(nombres_features), size=min(n_cuasi, n_features), replace=False)
    for idx in indices_cuasi:
        nombre = nombres_features[idx]
        valor_base = float(rng.uniform(0, 100))
        # Varianza muy baja: ruido minúsculo
        data[nombre] = valor_base + rng.uniform(-0.001, 0.001, size=n_filas)

    # Inyectar 1 par de features altamente correlacionadas
    if n_features >= 3:
        idx_base = int(rng.choice([i for i in range(n_features) if i not in indices_cuasi]))
        idx_corr = int(rng.choice([i for i in range(n_features)
                                    if i not in indices_cuasi and i != idx_base]))
        nombre_base = nombres_features[idx_base]
        nombre_corr = nombres_features[idx_corr]
        factor = float(rng.uniform(1.5, 5.0))
        offset = float(rng.uniform(-10, 10))
        ruido_minimo = rng.normal(0, 0.01, size=n_filas)
        data[nombre_corr] = data[nombre_base] * factor + offset + ruido_minimo

    # Inyectar nulos aleatorios en algunas columnas
    n_cols_con_nulos = int(rng.integers(1, min(4, n_features) + 1))
    cols_con_nulos = rng.choice(nombres_features, size=n_cols_con_nulos, replace=False)
    for col in cols_con_nulos:
        pct_nulos = float(rng.uniform(0.05, 0.35))  # 5% a 35% nulos
        mascara = rng.random(size=n_filas) < pct_nulos
        arr = data[col].copy()
        arr[mascara] = np.nan
        data[col] = arr

    # Columna target (sin nulos)
    data[target_col] = rng.uniform(0, 100, size=n_filas)

    df = pd.DataFrame(data)

    # --- Cálculo del output esperado ---

    # 1) Features numéricas (excluyendo target)
    features_num = [c for c in df.columns if c != target_col and df[c].dtype in ["float64", "int64"]]
    X = df[features_num]
    n_features_original = len(features_num)

    # 2) Porcentaje de nulos
    porcentaje_nulos = {}
    for col in features_num:
        pct = round(float(X[col].isna().mean() * 100), 2)
        porcentaje_nulos[col] = pct

    # 3) Features cuasi-constantes (rellenar NaN con mediana para VarianceThreshold)
    X_filled = X.copy()
    for col in X_filled.columns:
        mediana = X_filled[col].median()
        X_filled[col] = X_filled[col].fillna(mediana)

    selector = VarianceThreshold(threshold=umbral_varianza)
    selector.fit(X_filled)
    mascara_varianza = selector.get_support()
    features_cuasi_constantes = [features_num[i] for i in range(n_features_original)
                                  if not mascara_varianza[i]]

    # 4) Pares redundantes (correlación absoluta > umbral)
    corr_matrix = X_filled.corr().abs()
    pares_redundantes = []
    for i in range(len(features_num)):
        for j in range(i + 1, len(features_num)):
            val = corr_matrix.iloc[i, j]
            if val > umbral_correlacion:
                pares_redundantes.append((
                    features_num[i],
                    features_num[j],
                    round(float(val), 4)
                ))

    # 5) Conteos
    n_features_utiles = n_features_original - len(features_cuasi_constantes)

    output_esperado = {
        "porcentaje_nulos": porcentaje_nulos,
        "features_cuasi_constantes": features_cuasi_constantes,
        "pares_redundantes": pares_redundantes,
        "n_features_original": n_features_original,
        "n_features_utiles": n_features_utiles,
    }

    input_dict = {
        "df": df,
        "target_col": target_col,
        "umbral_varianza": umbral_varianza,
        "umbral_correlacion": umbral_correlacion,
    }

    return input_dict, output_esperado


# ==================== DEMOSTRACIÓN ====================
if __name__ == "__main__":
    for ejecucion in range(1, 3):
        print("=" * 70)
        print(f"  EJECUCIÓN {ejecucion}")
        print("=" * 70)
        inp, out = generar_caso_de_uso_diagnosticar_dataset()
        print(f"\n  target_col:          {inp['target_col']}")
        print(f"  umbral_varianza:     {inp['umbral_varianza']}")
        print(f"  umbral_correlacion:  {inp['umbral_correlacion']}")
        print(f"  shape del df:        {inp['df'].shape}")
        print(f"  columnas:            {list(inp['df'].columns)}")
        print(f"\n  DataFrame (primeras 5 filas):")
        print(inp["df"].head(5).to_string(index=False))
        print(f"\n  --- Output esperado ---")
        print(f"  porcentaje_nulos:          {out['porcentaje_nulos']}")
        print(f"  features_cuasi_constantes: {out['features_cuasi_constantes']}")
        print(f"  pares_redundantes:         {out['pares_redundantes']}")
        print(f"  n_features_original:       {out['n_features_original']}")
        print(f"  n_features_utiles:         {out['n_features_utiles']}")
        print()
