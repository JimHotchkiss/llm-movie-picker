import pandas as pd

df = pd.read_csv("./data/netflix_titles.csv")

print(f"df.columns: {df.columns}")
print(f"pd.isna(df): {pd.isna(df).sum()}")
print(f"df.info: {df.info}")

for description in df['description']:
    print(f"len(description): {len(description)}")
