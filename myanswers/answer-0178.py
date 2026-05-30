"""Limpiar y ordenar estudiantes.

Pipeline:
 - Deduplicación: Elimina registros repetidos por nombre conservando la primera aparición.
 - Ordenamiento: Ordena por nota de forma descendente (mayor a menor).

Dependencias: pandas, numpy.
"""


def limpiar_y_ordenar_estudiantes(df, nombre_col, nota_col):
    """
    Elimina duplicados por nombre y ordena por nota de forma descendente.
    """
    # 1. Eliminar duplicados manteniendo la primera aparición
    df_clean = df.drop_duplicates(subset=[nombre_col])
    # 2. Ordenar por nota de forma descendente
    df_sorted = df_clean.sort_values(by=nota_col, ascending=False)
    return df_sorted