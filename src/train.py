import pandas as pd
import pickle
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# receive the data path from the command line
if len(sys.argv) != 2:
    print("USAGE: python train.py <data_path>")
    sys.exit(1)

data_path = sys.argv[-1]
print(f"Loading data from {data_path}")

name_model = data_path.split("train-")[1].split(".")[0]

# load the data with parquet format
df = pd.read_parquet(data_path)

X = df.drop("total_sales", axis=1)
y = df["total_sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1912)

print("Training model...")

model = RandomForestRegressor(n_estimators=100, random_state=195)
model.fit(X_train, y_train)

# Specify the file path where you want to save the pickle file
file_path = f"../models/model-{name_model}.pkl"

# Save the model as a pickle file
with open(file_path, "wb") as f:
    pickle.dump(model, f)

print(f"Saving to {file_path} file...")

