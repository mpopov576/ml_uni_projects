import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('restaurant_clf.csv')

raw_audit = df.describe(include="all").T
raw_audit.to_csv('raw_audit.csv')

# columns drop

drop_cols = [
    "url",
    "address",
    "name",
    "phone",
    "dish_liked",
    "reviews_list",
    "menu_item",
    "listed_in(type)"
]

df = df.drop(columns=drop_cols)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# types transformation

type_map = {
    "online_order": "binary",
    "book_table": "binary",
    "votes": "int",
    "approx_cost(for two people)": "float",
    "location": "categorical",
    "rest_type": "categorical",
    "cuisines": "categorical",
    "listed_in(city)": "categorical"
}

# rate is transformed outside the other because special symbols

df["rate"] = pd.to_numeric(df["rate"].str.replace("/5", "", regex=False), errors="coerce")
median_rate = df["rate"].median()
df["rate"] = (df["rate"] > median_rate).astype(int)

for col, type in type_map.items():
    if type == "binary":
        df[col] = df[col].map({"Yes": 1, "No": 0})

    elif type == "float":
        df[col] = df[col].astype(str).str.replace(",", "")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    elif type == "int":
        df[col] = pd.to_numeric(df[col], errors="coerce")

    elif type == "categorical":
        df[col] = df[col].astype("category")

# removing null values

for col in df.columns:
    missing_ratio = df[col].isnull().sum() / len(df)

    # first check to not remove important feature

    if missing_ratio > 0.5:
        df.drop(columns=[col], inplace=True)

    elif df[col].dtype in ["int64", "float64"]:
        df[col] = df[col].fillna(df[col].mean())

    elif df[col].dtype == "category":
        df[col] = df[col].fillna(df[col].mode()[0])

processed_audit = df.describe(include="all").T
processed_audit.to_csv('processed_audit.csv')

df.to_csv('processed_restaurant_clf.csv')

# plotting

num_df = df.select_dtypes(include=["int64", "float64"])

# correlation matrix
corr = num_df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Correlation Matrix of Numeric Features", fontsize=12)
plt.show()

num_df = df.select_dtypes(include=["int64", "float64"])

sns.pairplot(num_df)

plt.show()

corr_table = df.corr(numeric_only=True)["rate"].drop("rate")
corr_table = corr_table.sort_values(ascending=False).reset_index()
corr_table.columns = ["Feature", "Correlation_with_rate"]

num_df = df.select_dtypes(include=["int64", "float64"])

num_df.hist(figsize=(12, 8), bins=30)

plt.suptitle("Numeric Feature Distributions", fontsize=14)
plt.tight_layout()
plt.show()