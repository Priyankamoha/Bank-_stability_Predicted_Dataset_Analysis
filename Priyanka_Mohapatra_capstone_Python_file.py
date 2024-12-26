import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

Bank_Stability=pd.read_csv(r'd:\ByteIQ\Caption project byteiq\Bank_Stability_Dataset.csv')

Bank_Stability.head()

Bank_Stability.info()

# Check for missing values
missing_values = Bank_Stability.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

# Handle Missing Values (Example)
numeric_columns = Bank_Stability.select_dtypes(include=np.number).columns
df = Bank_Stability.copy()  # Create a copy to avoid modifying the original DataFrame
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Check for duplicate rows
duplicates = Bank_Stability.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(Bank_Stability.describe())

# Visualizations
plt.figure(figsize=(10, 6))
numerical_features = Bank_Stability.select_dtypes(include=['number'])
sns.heatmap(numerical_features.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Pairplot for numerical data
sns.pairplot(Bank_Stability)
plt.show()

# Univariate Analysis
for column in Bank_Stability.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(Bank_Stability[column], kde=True, bins=30)
    plt.title(f"Distribution of {column}")
    plt.show()

# Encode Categorical Variables (if any)
categorical_columns = Bank_Stability.select_dtypes(include=['object']).columns
df = pd.get_dummies(Bank_Stability, columns=categorical_columns, drop_first=True)

# Check for outliers (Z-Score method)
from scipy.stats import zscore
z_scores = np.abs(zscore(Bank_Stability.select_dtypes(include=['float64', 'int64'])))
outliers = (z_scores > 3).any(axis=1)
print(f"\nOutliers detected: {outliers.sum()}")

# Remove outliers
df = Bank_Stability[~outliers]

# Save cleaned dataset
cleaned_file_path = r'\ByteIQ\Caption project byteiq/Cleaned_Bank_Stability_Dataset.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned dataset saved at {cleaned_file_path}")

"""**Model Selection and Training**"""

# Display column names
print("Columns in the dataset:")
print(Bank_Stability.columns)

# # Check for missing values in the features and target
# print(pd.DataFrame(X_train).isnull().sum())  # For features
# print(y_train.isnull().sum())  # For target

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define features and target
target_column = 'Went_Defunct'  # Replace with the correct column name for the target
features = df.drop(columns=[target_column])
target = df[target_column]

# Select only numerical features for scaling
numerical_features = features.select_dtypes(include=['number'])

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(numerical_features)

# Create a new DataFrame with scaled numerical features
features_scaled_df = pd.DataFrame(features_scaled, columns=numerical_features.columns, index=features.index)

# Concatenate scaled numerical features with original non-numerical features
features_final = pd.concat([features.select_dtypes(exclude=['number']), features_scaled_df], axis=1)

# Convert 'Bank_Name' column to numerical using Label Encoding before scaling
label_encoder = LabelEncoder()
features['Bank_Name_Encoded'] = label_encoder.fit_transform(features['Bank_Name'])
features = features.drop(columns=['Bank_Name']) # Drop original Bank_Name column

# Concatenate scaled numerical features with original non-numerical features
# NOTE: There should be no non-numerical features left after the above encoding
features_final = features_scaled_df

# Split data
X_train, X_test, y_train, y_test = train_test_split(features_final, target, test_size=0.1, random_state=42)

# Encode the target variable using LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Identify columns with categorical data (string type)
categorical_columns = X_train.select_dtypes(include=['object']).columns

# Apply One-Hot Encoding for categorical columns
X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

# Ensure both X_train_encoded and X_test_encoded have the same columns
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

y_train = y_train.fillna(y_train.mode()[0])

# Now, you can safely train the model
# model.fit(X_train_encoded, y_train)

# Train a Random Forest Classifier using the encoded target variable
model = RandomForestClassifier(random_state=45)
model.fit(X_train, y_train_encoded)  

# Evaluate the model using the encoded target variable
y_pred_encoded = model.predict(X_test)  
y_pred = label_encoder.inverse_transform(y_pred_encoded) 
print(f"Accuracy: {accuracy_score(y_test_encoded, y_pred_encoded)}")
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_encoded))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred_encoded))

# Save the model
import joblib
model_path = r'\ByteIQ\Caption project byteiq/Bank_Stability_Model.pkl'
joblib.dump(model, model_path)
print(f"Model saved at {model_path}")

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Step 4: Train Logistic Regression Model
model = LogisticRegression(random_state=42, max_iter=1000)  
model.fit(X_train, y_train_encoded)

# Step 5: Evaluate the Model
# Make predictions
y_pred_encoded = model.predict(X_test)  
y_pred = label_encoder.inverse_transform(y_pred_encoded)  

# Print metrics
print(f"Accuracy: {accuracy_score(y_test_encoded, y_pred_encoded)}")
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_encoded))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred_encoded))

from sklearn.tree import DecisionTreeClassifier
# Train a Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions and encode them
y_pred_encoded = model.predict(X_test)
y_pred_encoded = label_encoder.transform(y_pred_encoded) 

# Calculate accuracy
print(f"Accuracy: {accuracy_score(y_test_encoded, y_pred_encoded):.2f}")

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_encoded))

# Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred_encoded))

# Step 6: Visualize the Decision Tree 
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Get the target column name from y_train
target_column_name = y_train.name

# Get unique class names from the target variable
class_names = y_train.unique().tolist() 

# **Create a new DecisionTreeClassifier object**
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train) 

# Plot the Decision Tree using the DecisionTreeClassifier object
plt.figure(figsize=(12, 8))  
plot_tree(decision_tree_model, feature_names=list(X_train.columns),  
          class_names=class_names,  
          filled=True, rounded=True)
          
plt.title("Decision Tree Visualization")
plt.show()

"""**GradientBoostingClassfier**"""

from sklearn.ensemble import GradientBoostingClassifier

# Step 4: Train a Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# Make predictions and encode them
y_pred_encoded = model.predict(X_test)
y_pred_encoded = label_encoder.transform(y_pred_encoded)  # Encode predictions to match y_test_encoded

# Calculate accuracy
print(f"Accuracy: {accuracy_score(y_test_encoded, y_pred_encoded):.2f}")

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_encoded))

# Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred_encoded))

# Step 6: Feature Importance (Optional)
import matplotlib.pyplot as plt
import numpy as np

# Plot feature importances
feature_importances = model.feature_importances_
sorted_idx = np.argsort(feature_importances)

plt.figure(figsize=(10, 6))
plt.barh(X_train.columns[sorted_idx], feature_importances[sorted_idx], color="skyblue") # Changed X to X_train
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Gradient Boosting Feature Importance")
plt.show()

# Step 7: Save the Model (Optional)
import joblib
model_path = 'd:\ByteIQ\Caption project byteiq\Gradient_Boosting_Model.pkl'
joblib.dump(model, model_path)
print(f"Gradient Boosting Model saved at {model_path}")

"""XGBoost"""

from xgboost import XGBClassifier

# Step 4: Train XGBoost Classifier
model = XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.01, max_depth=3)
model.fit(X_train, y_train_encoded)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy (XGBoost): {accuracy_score(y_test_encoded, y_pred_encoded):.2f}")
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_encoded))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred_encoded))

import joblib
model_path = 'd:\ByteIQ\Caption project byteiq\XGBoosting_Model.pkl'
joblib.dump(model, model_path)
print(f"XGBoosting Model saved at {model_path}")

# from sklearn.ensemble import VotingClassifier

# # Train the Gradient Boosting model
# gb_model = GradientBoostingClassifier(random_state=42)
# gb_model.fit(X_train, y_train)

# # Train the XGBoost model
# xgb_model = XGBClassifier(random_state=42)
# xgb_model.fit(X_train, y_train)

# # Combine models into a Voting Classifier
# combined_model = VotingClassifier(
#     estimators=[
#         ('gb', gb_model),
#         ('xgb', xgb_model)
#     ],
#     voting='soft'
# )

# # Fit the VotingClassifier on training data
# combined_model.fit(X_train, y_train)

# # Combine models into a Voting Classifier
# combined_model = VotingClassifier(
#     estimators=[
#         ('gb', model),
#         ('xgb', model)
#     ],
#     voting='soft'  # 'soft' uses predicted probabilities
# )

# # Fit the VotingClassifier on training data
# combined_model.fit(X_train, y_train)

# import joblib

# # Save individual models
# joblib.dump(model, 'Gradient_Boosting_Model.pkl')
# joblib.dump(model, 'XGBoosting_Model.pkl')

# # Save combined model
# joblib.dump(combined_model, 'Combined_Model.pkl')
