"""Resumir estadísticas por grupo.

Pipeline:
 - Agrupación: Agrupa los registros por col_grupo.
 - Agregación: Calcula media, mediana, desviación estándar y conteo de col_valor.
 - Ordenamiento: Ordena por media de forma descendente (mayor a menor).
 - Reindexado: Reinicia el índice del resultado.

Dependencias: pandas.
"""


def resumir_por_grupo(df, col_grupo, col_valor):
    """
    Agrupa un DataFrame por col_grupo y calcula estadísticas para col_valor.
    Retorna un DataFrame ordenado por media de mayor a menor con índice reiniciado.
    """
    # 1. Agrupar y 2. Calcular estadísticas con nombres de columnas específicos
    resultado = df.groupby(col_grupo)[col_valor].agg(
        media='mean',
        mediana='median',
        std='std',
        conteo='count'
    )
    
    # 3. Ordenar por 'media' descendente y 4. Reiniciar índice
    resultado = resultado.sort_values('media', ascending=False).reset_index(drop=True)
    
    return resultado