import pandas as pd
import numpy as np


def generar_caso_de_uso_construir_coocurrencia():
    """
    Genera un par (input, output) aleatorio para la función construir_coocurrencia.

    Returns:
        tuple: (input_dict, output_esperado)
            - input_dict: dict con claves 'df', 'transaccion_col', 'producto_col'
            - output_esperado: pd.DataFrame cuadrado y simétrico con co-ocurrencias,
              filas y columnas = productos ordenados alfabéticamente, diagonal = 0.
    """
    rng = np.random.default_rng()

    # --- Catálogo de productos aleatorio ---
    catalogo_base = [
        "Laptop", "Mouse", "Teclado", "Monitor", "Auriculares", "Webcam",
        "Cable_HDMI", "Hub_USB", "Cargador", "Mochila", "Tablet", "SSD",
        "Memoria_RAM", "Mousepad", "Soporte", "Lampara", "Silla", "Micrófono"
    ]
    n_productos = int(rng.integers(4, 9))  # 4 a 8 productos en el catálogo
    productos = sorted(rng.choice(catalogo_base, size=n_productos, replace=False).tolist())

    # --- Generar transacciones aleatorias ---
    n_transacciones = int(rng.integers(8, 21))  # 8 a 20 transacciones

    registros = []
    for t in range(1, n_transacciones + 1):
        id_transaccion = f"TXN-{str(t).zfill(3)}"
        # Cada transacción tiene entre 2 y min(5, n_productos) productos
        k = int(rng.integers(2, min(5, n_productos) + 1))
        prods_comprados = rng.choice(productos, size=k, replace=False).tolist()
        for p in prods_comprados:
            registros.append({"id_txn": id_transaccion, "producto": p})

    df = pd.DataFrame(registros)

    # Nombres de columnas aleatorios
    nombres_txn = rng.choice(["id_txn", "transaccion_id", "orden_id", "ticket"])
    nombres_prod = rng.choice(["producto", "item", "nombre_producto", "articulo"])
    df = df.rename(columns={"id_txn": nombres_txn, "producto": nombres_prod})

    # --- Cálculo del output esperado ---
    # Self-merge por columna de transacción
    merged = df.merge(df, on=nombres_txn, suffixes=("_a", "_b"))

    col_a = f"{nombres_prod}_a"
    col_b = f"{nombres_prod}_b"

    # Eliminar pares consigo mismo
    merged = merged[merged[col_a] != merged[col_b]]

    # Contar co-ocurrencias por par de productos
    conteo = merged.groupby([col_a, col_b])[nombres_txn].nunique().reset_index()
    conteo.columns = ["producto_a", "producto_b", "coocurrencias"]

    # Construir matriz pivote
    matriz = conteo.pivot(
        index="producto_a", columns="producto_b", values="coocurrencias"
    ).fillna(0).astype(int)

    # Asegurar que todos los productos estén presentes (filas y columnas)
    todos = sorted(df[nombres_prod].unique())
    matriz = matriz.reindex(index=todos, columns=todos, fill_value=0)

    # Diagonal en cero (por si acaso)
    vals = matriz.values.copy()
    np.fill_diagonal(vals, 0)
    matriz = pd.DataFrame(vals, index=matriz.index, columns=matriz.columns)

    matriz.index.name = None
    matriz.columns.name = None

    output_esperado = matriz

    input_dict = {
        "df": df,
        "transaccion_col": nombres_txn,
        "producto_col": nombres_prod,
    }

    return input_dict, output_esperado


# ==================== DEMOSTRACIÓN ====================
if __name__ == "__main__":
    for ejecucion in range(1, 3):
        print("=" * 70)
        print(f"  EJECUCIÓN {ejecucion}")
        print("=" * 70)
        inp, out = generar_caso_de_uso_construir_coocurrencia()
        print(f"\n  transaccion_col: {inp['transaccion_col']}")
        print(f"  producto_col:    {inp['producto_col']}")
        print(f"  filas en df:     {len(inp['df'])}")
        print(f"  transacciones:   {inp['df'][inp['transaccion_col']].nunique()}")
        print(f"  productos:       {sorted(inp['df'][inp['producto_col']].unique())}")
        print(f"\n  DataFrame de entrada (primeras 12 filas):")
        print(inp["df"].head(12).to_string(index=False))
        print(f"\n  Matriz de co-ocurrencia ({out.shape[0]}x{out.shape[1]}):")
        print(out.to_string())
        # Verificar simetría
        es_simetrica = (out.values == out.values.T).all()
        diagonal_cero = (np.diag(out.values) == 0).all()
        print(f"\n  ¿Simétrica? {es_simetrica}  |  ¿Diagonal = 0? {diagonal_cero}")
        print()
