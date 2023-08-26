import pickle
import pandas as pd
import sys

path_model = sys.argv[-2]
path_data = sys.argv[-1]

model = pickle.load(open(path_model, "rb"))

# Carrega o arquivo do bank_predict.csv 
df = pd.read_parquet(path_data)
X_test = df  # nao precisa dropar a coluna total_sales pq ela nao existe no predict

print(f"Loading data from {path_data}")
print(f"Loading model from {path_model}")

y_pred = model.predict(X_test)

print("Making predictions...")

# Adiciona uma nova coluna 'y_pred' com as predições 
df['y_pred'] = y_pred

print(df.head(20))

# Salva o DataFrame atualizado de novo no CSV file
df.to_parquet("../data/predict-done-2023-08-03.parquet", index=False)



