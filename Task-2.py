import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')

print("First 5 rows of the dataset:")
print(titanic.head())

print("\nSummary of missing values:")
print(titanic.isnull().sum())

titanic['age'].fillna(titanic['age'].median(), inplace=True)
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

titanic.drop(columns=['deck'], inplace=True)

print("\nUpdated summary of missing values:")
print(titanic.isnull().sum())

print("\nSummary statistics:")
print(titanic.describe())

plt.figure(figsize=(10, 5))
sns.histplot(titanic['age'], kde=True, bins=30, color='blue', alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='survived', data=titanic, palette='Set2')
plt.title('Survival Count')
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x='sex', y='survived', data=titanic, palette='Set3')
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x='pclass', y='survived', data=titanic, palette='Set1')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

plt.figure(figsize=(10, 8))
corr = titanic.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

sns.pairplot(titanic, hue='survived', diag_kind='kde', palette='husl')
plt.show()

print("\nInsights from the analysis:")
print("- Younger passengers and women had higher survival rates.")
print("- First-class passengers had a significantly higher survival rate.")
print("- Survival rate shows correlation with age, gender, and class.")
