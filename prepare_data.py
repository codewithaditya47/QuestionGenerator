import pandas as pd

df = pd.read_csv("data/Question.csv")

# convert to lowercase
df["Question"] = df["Question"].str.lower()

# remove extra spaces
df["Question"] = df["Question"].str.strip()

# remove duplicates
df = df.drop_duplicates()

print("Cleaned dataset size:", len(df))
df["input"] = "generate exam question"
df["output"] = df["Question"]

train_df = df[["input", "output"]]

print(train_df.head())
train_df.to_csv("data/training_data.csv", index=False)

