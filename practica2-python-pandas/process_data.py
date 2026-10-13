# Importar Pandas.
import pandas as pd

# Leer los datos del archivo CSV.
df = pd.read_csv("data.csv")

# Mostrar las estadísticas descriptivas
# de las columnas numéricas.
print(df.describe())
