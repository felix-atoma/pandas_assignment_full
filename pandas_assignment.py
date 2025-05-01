import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("First few rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData description:")
print(df.describe())

print("\nAverage petal length per species:")
print(df.groupby('species')['petal length (cm)'].mean())

# Line Chart
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.title('Sepal Length Trend over Samples')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.tight_layout()
plt.savefig('line_chart.png')
plt.close()

# Bar Chart
plt.figure(figsize=(8, 6))
df.groupby('species')['petal length (cm)'].mean().plot(kind='bar', color='skyblue')
plt.title('Average Petal Length by Species')
plt.ylabel('Petal Length (cm)')
plt.xlabel('Species')
plt.tight_layout()
plt.savefig('bar_chart.png')
plt.close()

# Histogram
plt.figure(figsize=(8, 6))
plt.hist(df['petal width (cm)'], bins=20, color='lightgreen')
plt.title('Distribution of Petal Width')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('histogram.png')
plt.close()

# Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs. Petal Length')
plt.tight_layout()
plt.savefig('scatter_plot.png')
plt.close()
