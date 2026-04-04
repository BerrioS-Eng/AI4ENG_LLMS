import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


def generar_caso_de_uso_seleccionar_gmm_optimo():
    """
    Genera un par (input, output) aleatorio para la función seleccionar_gmm_optimo.

    Returns:
        tuple: (input_dict, output_esperado)
            - input_dict: dict con claves 'X', 'rango_componentes', 'random_state'
            - output_esperado: dict con claves 'mejor_k', 'mejor_bic',
              'bic_por_k', 'etiquetas', 'probabilidades'
    """
    rng = np.random.default_rng()

    # --- Parámetros aleatorios ---
    random_state = int(rng.integers(0, 1000))
    n_features = int(rng.integers(2, 6))          # 2 a 5 features
    k_real = int(rng.integers(2, 5))              # 2 a 4 clusters reales
    n_samples_por_cluster = rng.integers(30, 71, size=k_real)  # 30-70 por cluster

    # Generar datos con clusters gaussianos bien separados
    bloques = []
    for i in range(k_real):
        centro = rng.uniform(-10, 10, size=n_features)
        dispersion = rng.uniform(0.3, 1.5)
        bloque = rng.normal(loc=centro, scale=dispersion,
                            size=(int(n_samples_por_cluster[i]), n_features))
        bloques.append(bloque)

    X = np.vstack(bloques)

    # Barajar filas para no tener orden por cluster
    indices = rng.permutation(len(X))
    X = X[indices]

    # Rango de componentes: incluye el real y algunos más
    k_min = 2
    k_max = int(rng.integers(k_real + 1, k_real + 4))  # hasta 3 por encima del real
    rango_componentes = list(range(k_min, k_max + 1))

    # --- Cálculo del output esperado ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    bic_por_k = {}
    modelos = {}
    for k in rango_componentes:
        gmm = GaussianMixture(n_components=k, random_state=random_state)
        gmm.fit(X_scaled)
        bic_por_k[k] = gmm.bic(X_scaled)
        modelos[k] = gmm

    # Seleccionar k con menor BIC; en empate, el más pequeño
    mejor_k = min(bic_por_k, key=lambda k: (bic_por_k[k], k))
    mejor_bic = bic_por_k[mejor_k]

    modelo_optimo = modelos[mejor_k]
    etiquetas = modelo_optimo.predict(X_scaled)
    probabilidades = modelo_optimo.predict_proba(X_scaled)

    output_esperado = {
        "mejor_k": mejor_k,
        "mejor_bic": mejor_bic,
        "bic_por_k": bic_por_k,
        "etiquetas": etiquetas,
        "probabilidades": probabilidades,
    }

    input_dict = {
        "X": X,
        "rango_componentes": rango_componentes,
        "random_state": random_state,
    }

    return input_dict, output_esperado


# ==================== DEMOSTRACIÓN ====================
if __name__ == "__main__":
    for ejecucion in range(1, 3):
        print("=" * 70)
        print(f"  EJECUCIÓN {ejecucion}")
        print("=" * 70)
        inp, out = generar_caso_de_uso_seleccionar_gmm_optimo()
        print(f"\n  X shape:            {inp['X'].shape}")
        print(f"  rango_componentes:  {inp['rango_componentes']}")
        print(f"  random_state:       {inp['random_state']}")
        print(f"\n  X (primeras 5 filas):")
        print(np.array2string(inp["X"][:5], precision=3, suppress_small=True))
        print(f"\n  --- Output esperado ---")
        print(f"  mejor_k:       {out['mejor_k']}")
        print(f"  mejor_bic:     {out['mejor_bic']:.2f}")
        print(f"  bic_por_k:     {out['bic_por_k']}")
        print(f"  etiquetas:     {out['etiquetas'][:15]} ...")
        print(f"  probabilidades shape: {out['probabilidades'].shape}")
        print(f"  probabilidades (primeras 3 filas):")
        print(np.array2string(out["probabilidades"][:3], precision=4, suppress_small=True))
        print()
