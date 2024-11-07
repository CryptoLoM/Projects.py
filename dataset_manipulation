import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

# Load the dataset
file_path = 'top_10000_1950-now (1).xlsx - top_10000_1950-now.csv'
data = pd.read_csv(file_path)

# Display the first few rows and basic information about the dataset
data.head(), data.info()

# Retry by attempting to load the file as a CSV, given the filename's dual nature suggesting it might be in CSV format.
file_path_csv = 'top_10000_1950-now (1).xlsx - top_10000_1950-now.csv'
data = pd.read_csv(file_path_csv, encoding='utf-8')

# Display basic information to understand the structure and presence of missing values.
data.info(), data.head()

# Define a function to remove outliers based on the Z-score

# Drop column with no values
data_cleaned = data.drop(columns=["Album Genres"])

# Drop rows with any missing values
data_cleaned = data_cleaned.dropna()

# Confirm the dataset shape after cleaning and re-check info
data_cleaned.shape, data_cleaned.info()



# Define a function to remove outliers based on the Z-score
def remove_outliers(df, columns, threshold=3):
    # For each column, calculate the Z-score and filter based on threshold
    for col in columns:
        df = df[(np.abs((df[col] - df[col].mean()) / df[col].std()) <= threshold)]
    return df

# Columns to check for outliers
numeric_columns = [
    "Popularity", "Danceability", "Energy", "Loudness", "Speechiness",
    "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo"
]

# Remove outliers
data_cleaned = remove_outliers(data_cleaned, numeric_columns)

# Confirm shape after outlier removal
data_cleaned.shape
data_cleaned.to_csv("cleaned_dataset.csv", index=False)

data = pd.read_csv("cleaned_dataset.csv")

plt.figure(figsize=(10, 6))
sns.histplot(data['Popularity'], bins=30, kde=True, color="skyblue")
plt.title("Popularity Distribution")
plt.xlabel("Popularity")
plt.ylabel("Frequency")
plt.show()

# Apply Min-Max scaling
scaler = MinMaxScaler()
min_max_scaled = scaler.fit_transform(data[['Popularity', 'Danceability', 'Energy', 'Tempo']])

# Convert to DataFrame for plotting
min_max_df = pd.DataFrame(min_max_scaled, columns=['Popularity', 'Danceability', 'Energy', 'Tempo'])

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=min_max_df)
plt.title("Min-Max Normalized Values")
plt.show()


# Apply Z-score normalization
scaler = StandardScaler()
z_score_scaled = scaler.fit_transform(data[['Popularity', 'Danceability', 'Energy', 'Tempo']])

# Convert to DataFrame for plotting
z_score_df = pd.DataFrame(z_score_scaled, columns=['Popularity', 'Danceability', 'Energy', 'Tempo'])

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=z_score_df)
plt.title("Z-score Normalized Values")
plt.show()
