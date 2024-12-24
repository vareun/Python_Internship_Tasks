import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

categories = ['Male', 'Female', 'Other']
values = [300, 450, 50]

ages = np.random.normal(30, 10, 1000)  # 1000 random ages with mean=30, std=10

plt.figure(figsize=(8, 5))
plt.bar(categories, values, color=['blue', 'pink', 'green'])
plt.title('Distribution of Genders', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(ages, bins=30, kde=True, color='purple', alpha=0.7)
plt.title('Distribution of Ages', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

