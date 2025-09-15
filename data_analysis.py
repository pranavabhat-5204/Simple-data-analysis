import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (replace with your Titanic CSV path)
df = pd.read_csv("/content/train.csv")

# Set style
sns.set(style="whitegrid")

# 1. Survival count
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Survived", palette="Set2")
plt.title("Survival Count")
plt.show()

# 2. Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Sex", hue="Survived", palette="Set1")
plt.title("Survival by Gender")
plt.show()

# 3. Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Pclass", hue="Survived", palette="muted")
plt.title("Survival by Passenger Class")
plt.show()

# 4. Age distribution with KDE
plt.figure(figsize=(8,5))
sns.kdeplot(data=df, x="Age", hue="Survived", fill=True, common_norm=False, palette="coolwarm")
plt.title("Age Distribution by Survival")
plt.show()

# 5. Fare distribution by class (Violin plot)
plt.figure(figsize=(8,5))
sns.violinplot(data=df, x="Pclass", y="Fare", palette="Pastel1")
plt.title("Fare Distribution by Passenger Class")
plt.show()

# 6. Boxplot: Age by Class & Survival
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Pclass", y="Age", hue="Survived", palette="Set3")
plt.title("Age Distribution by Class and Survival")
plt.show()
