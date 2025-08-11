import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train.csv")

# Basic Inspection
print("=== Data Info ===")
print(df.info())
print("\n=== Summary Statistics ===")
print(df.describe())
print("\n=== Value counts for Sex ===")
print(df['Sex'].value_counts())

# Missing Values
print("\n=== Missing Values ===")
print(df.isnull().sum())

# 1. Age Distribution
plt.figure(figsize=(8,5))
df['Age'].hist(bins=30, edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# 2. Age by Passenger Class
plt.figure(figsize=(8,5))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title("Age by Passenger Class")
plt.show()


# 3. Survival by Sex
plt.figure(figsize=(6,5))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival Count by Sex")
plt.show()


# 4. Survival by Embarked
plt.figure(figsize=(6,5))
sns.countplot(x='Embarked', hue='Survived', data=df)
plt.title("Survival by Embarked Port")
plt.show()


# 5. Fare Distribution
plt.figure(figsize=(8,5))
df['Fare'].hist(bins=40, edgecolor='black')
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()


# 6. Scatterplot Age vs Fare
plt.figure(figsize=(8,5))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title("Age vs Fare by Survival")
plt.show()


# 7. Pairplot (selected variables)
sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']], hue='Survived')
plt.show()


# 8. Correlation Heatmap
corr = df.select_dtypes(include=['number']).corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()



