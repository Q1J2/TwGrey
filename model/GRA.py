import numpy as np
import pandas as pd

excel_file = r"D:\UCM\UCM100\100features.xlsx"
df = pd.read_excel(excel_file)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

rho = 0.4
epsilon = 1


grey_relation_degrees = np.zeros(X.shape[1])


differences = np.abs(X - y.reshape(-1, 1))
delta_min = differences.min()
delta_max = differences.max()


for i in range(X.shape[1]):
    grey_relation_degrees[i] = np.mean((delta_min + rho * delta_max) / (differences[:, i] + rho * delta_max))

feature_names = df.columns[:-1]
for i, degree in enumerate(grey_relation_degrees):
    print(f"Feature: {feature_names[i]}, Grey Relational Degree: {degree:.4f}")
output_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance_Score': grey_relation_degrees
})


output_df = output_df.sort_values(by='Importance_Score', ascending=False)

output_path = r"D:\UCM\UCM100\grey_0.4.csv"
output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\nSaved grey relational degrees to: {output_path}")
