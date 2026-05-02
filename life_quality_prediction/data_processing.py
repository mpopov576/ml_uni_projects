import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("lifestyle_in_different_towns.csv")

df_raw_describe = df.describe(include="all").T

df_raw_describe.to_csv("raw_audit.csv")

# no missing values were detected
# print(df.isnull().sum())

df = df.drop(columns=["city_name"])

df = pd.get_dummies(df, columns=["country"], drop_first=True)


# its ok all the values to be floats because the scaling after makes the floats anyway
df = df.astype(float)

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
# print(outliers)

df_capped = df.copy()
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df_capped = df_capped.clip(lower, upper, axis=1)

df = df_capped

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled = df_scaled.loc[:, df_scaled.std() != 0]

df_scaled.to_csv("processed_data.csv")


# df_processed_audit = df_scaled.describe(include="all").T
# df_processed_audit.to_csv("processed_audit.csv")
#
# df_scaled_num = df_scaled.select_dtypes(include=[np.number])
#
# cols = ["avg_income", "avg_rent", "happiness_score", "air_quality_index"]
# sns.pairplot(df_scaled_num[cols])
# plt.show()
#
# plt.figure(figsize=(12, 6))
# sns.heatmap(df_scaled_num[cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Matrix")
# plt.show()
#
# df_scaled_num.hist(figsize=(12, 8), bins=20)
# plt.suptitle("Distributions of Scaled Numeric Features")
# plt.show()
