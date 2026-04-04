import pandas as pd
import numpy as np


def generar_caso_de_uso_analizar_etiquetas():
    """
    Genera un par (input, output) aleatorio para la función analizar_etiquetas.

    Returns:
        tuple: (input_dict, output_esperado)
            - input_dict: dict con claves 'df', 'categoria_col', 'etiquetas_col'
            - output_esperado: tupla (frecuencia_global, resumen_por_categoria)
              - frecuencia_global: pd.Series (índice=etiqueta, valor=conteo, desc)
              - resumen_por_categoria: pd.DataFrame con columnas
                ["categoria", "total_etiquetas", "etiquetas_unicas", "etiqueta_top"],
                ordenado alfabéticamente por categoria, índice reiniciado.
    """
    rng = np.random.default_rng()

    # --- Catálogo de etiquetas y categorías ---
    pool_etiquetas = [
        "python", "machine_learning", "estadistica", "deep_learning",
        "sql", "visualizacion", "nlp", "big_data", "cloud", "docker",
        "git", "algebra_lineal", "probabilidad", "redes_neuronales",
        "web_scraping", "api_rest", "pandas", "tensorflow", "scikit_learn",
        "optimizacion", "series_temporales", "computer_vision"
    ]
    pool_categorias = [
        "Programación", "Data Science", "Ingeniería de Datos",
        "Inteligencia Artificial", "Matemáticas", "DevOps"
    ]

    n_categorias = int(rng.integers(3, 6))  # 3 a 5 categorías
    categorias = sorted(rng.choice(pool_categorias, size=n_categorias, replace=False).tolist())

    n_cursos = int(rng.integers(12, 31))  # 12 a 30 cursos

    # --- Generar cursos con etiquetas ---
    registros = []
    for i in range(n_cursos):
        cat = str(rng.choice(categorias))
        n_tags = int(rng.integers(1, 6))  # 1 a 5 etiquetas por curso
        tags = rng.choice(pool_etiquetas, size=n_tags, replace=False).tolist()
        # Introducir espacios aleatorios para probar el .strip()
        tags_con_ruido = []
        for t in tags:
            espacios = rng.choice(["", " ", "  "])
            tags_con_ruido.append(espacios + t + espacios)
        etiquetas_str = ",".join(tags_con_ruido)
        registros.append({"cat": cat, "tags": etiquetas_str})

    df = pd.DataFrame(registros)

    # Nombres de columnas aleatorios
    nombres_cat = str(rng.choice(["categoria", "area", "tipo_curso", "departamento"]))
    nombres_tags = str(rng.choice(["etiquetas", "tags", "temas", "keywords"]))
    df = df.rename(columns={"cat": nombres_cat, "tags": nombres_tags})

    # --- Cálculo del output esperado ---

    # Explotar etiquetas
    df_calc = df.copy()
    df_calc[nombres_tags] = df_calc[nombres_tags].str.split(",")
    df_exp = df_calc.explode(nombres_tags)
    df_exp[nombres_tags] = df_exp[nombres_tags].str.strip()

    # 1) Frecuencia global
    frecuencia_global = (
        df_exp[nombres_tags]
        .value_counts()
        .sort_values(ascending=False)
    )
    frecuencia_global.name = None

    # 2) Resumen por categoría
    # Total de etiquetas por categoría (contando repeticiones)
    total = df_exp.groupby(nombres_cat)[nombres_tags].count().rename("total_etiquetas")

    # Etiquetas únicas por categoría
    unicas = df_exp.groupby(nombres_cat)[nombres_tags].nunique().rename("etiquetas_unicas")

    # Etiqueta más frecuente (moda) por categoría
    def obtener_top(grupo):
        return grupo.value_counts().index[0]

    top = df_exp.groupby(nombres_cat)[nombres_tags].apply(obtener_top).rename("etiqueta_top")

    resumen = pd.concat([total, unicas, top], axis=1).reset_index()
    resumen = resumen.rename(columns={nombres_cat: "categoria"})
    resumen = resumen.sort_values("categoria").reset_index(drop=True)
    resumen = resumen[["categoria", "total_etiquetas", "etiquetas_unicas", "etiqueta_top"]]

    output_esperado = (frecuencia_global, resumen)

    input_dict = {
        "df": df,
        "categoria_col": nombres_cat,
        "etiquetas_col": nombres_tags,
    }

    return input_dict, output_esperado


# ==================== DEMOSTRACIÓN ====================
if __name__ == "__main__":
    for ejecucion in range(1, 3):
        print("=" * 70)
        print(f"  EJECUCIÓN {ejecucion}")
        print("=" * 70)
        inp, out = generar_caso_de_uso_analizar_etiquetas()
        freq, resumen = out
        print(f"\n  categoria_col:  {inp['categoria_col']}")
        print(f"  etiquetas_col:  {inp['etiquetas_col']}")
        print(f"  filas en df:    {len(inp['df'])}")
        print(f"\n  DataFrame de entrada (primeras 8 filas):")
        print(inp["df"].head(8).to_string(index=False))
        print(f"\n  --- Output esperado ---")
        print(f"\n  Frecuencia global (top 10):")
        print(freq.head(10).to_string())
        print(f"\n  Resumen por categoría:")
        print(resumen.to_string(index=True))
        print()
