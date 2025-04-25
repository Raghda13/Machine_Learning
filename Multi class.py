import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(r"C:\Users\Pc Store\Downloads\teleCust1000t.csv")
print("Columns in dataset:", df.columns.tolist())

# Ensure target column exists
target_column = 'custcat'
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' is missing. Check your dataset.")

# Visualize the distribution of the target variable
sns.countplot(y=target_column, data=df)
plt.title(f'Distribution of {target_column}')
plt.show()

# Check for null values and dataset summary
print(df.info())
print(df.describe())
print("Missing values per column:\n", df.isnull().sum())

# Standardizing continuous numerical features
continuous_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
continuous_columns.remove(target_column)  # Exclude target column
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[continuous_columns])
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
scaled_data = pd.concat([df.drop(columns=continuous_columns), scaled_df], axis=1)

# One-hot encoding categorical variables
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

# Encode the target variable
prepped_data[target_column] = prepped_data[target_column].astype('category').cat.codes

# Separate input and target data
X = prepped_data.drop(target_column, axis=1)
y = prepped_data[target_column]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train and evaluate logistic regression (One-vs-All)
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)
y_pred_ova = model_ova.predict(X_test)
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100 * accuracy_score(y_test, y_pred_ova), 2)}%")

# Train and evaluate logistic regression (One-vs-One)
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)
y_pred_ovo = model_ovo.predict(X_test)
print("One-vs-One (OvO) Strategy")
print(f"Accuracy: {np.round(100 * accuracy_score(y_test, y_pred_ovo), 2)}%")
