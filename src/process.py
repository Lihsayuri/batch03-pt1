import pandas as pd
import datetime


df_original = pd.read_csv("../data/train-2023-08-01.csv")

df = df_original.copy()

# group prices by date abd store_id and create a new column with the sum of prices named total_sales
df = df.groupby(["date", "store_id"]).agg({"price": "sum"}).reset_index()
df = df.rename(columns={"price": "total_sales"})

# separe a coluna date em 3 com o ano, mes e dia
df["year"] = df["date"].apply(lambda x: x.split("-")[0])
df["month"] = df["date"].apply(lambda x: x.split("-")[1])
df["day"] = df["date"].apply(lambda x: x.split("-")[2])

# cria a coluna weekday com o dia da semana, sendo de 0 a 6, onde 0 é segunda e 6 é domingo
df["weekday"] = df["date"].apply(
    lambda dateString: datetime.datetime.strptime(dateString, "%Y-%m-%d").weekday()
)


# drop the date column
df = df.drop("date", axis=1)

# save 

df.to_csv("../data/data-train-2023-08-01-processed", index=False)

# salva os dados em formato parquet
df.to_parquet("../data/data-train-2023-08-01-processed.parquet", index=False)
