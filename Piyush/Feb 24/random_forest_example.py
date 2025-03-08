# Import necessary libraries for data analysis and machine learning
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical operations
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # For splitting data and tuning model
from sklearn.ensemble import RandomForestClassifier  # Our main prediction algorithm
from sklearn.pipeline import Pipeline  # For creating sequences of data processing steps
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # For data normalization and encoding
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.compose import ColumnTransformer  # For applying different transformations to different columns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score  # For model evaluation

# 1. Load dataset
# Reading the CSV file into memory as a table-like structure (DataFrame)
df = pd.read_csv('./data.csv')
print("Dataset loaded with shape:", df.shape)  # Shows number of rows and columns in our data

# 2. Define features dynamically based on data types
# This automatically categorizes our data columns instead of manually specifying them
NUMERICAL_FEATURES = []  # Will hold columns containing numbers (like age, weight, etc.)
CATEGORICAL_FEATURES = []  # Will hold columns containing text/categories (like gender, color, etc.)

# Loop through each column in our data and categorize it
for name in df.columns:
    if df[name].dtype == 'object':  # If column contains text/strings
        CATEGORICAL_FEATURES.append(name)
    else:  # If column contains numbers
        NUMERICAL_FEATURES.append(name)

# Remove the outcome we're trying to predict from our feature list
# We don't want to include what we're predicting as an input feature!
target_variable = 'Outcome Variable'  # The column containing what we want to predict
if target_variable in CATEGORICAL_FEATURES:
    CATEGORICAL_FEATURES.remove(target_variable)

# Combine all our input features into one list
FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
print("Features detected:", FEATURES)  # Show which columns will be used for prediction

# 3. Set up data preprocessing pipelines
# Different types of data need different preparation techniques

# For numerical data:
# 1) Fill in missing values with median (middle value) - better than average when there are outliers
# 2) Scale values to similar ranges so no feature dominates the model unfairly
numeric_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),  # Replace missing values with median
    ('scaler', StandardScaler())  # Adjust values to have mean=0 and standard deviation=1
])

# For categorical data:
# 1) Fill in missing values with the word "missing" 
# 2) Convert text categories to numbers using one-hot encoding (creates separate columns for each category)
categorical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant', fill_value='missing')),  # Replace missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Convert text to numbers
])

# 4. Combine the preprocessing steps for all data types
# This creates one unified preprocessing system that handles all our data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, NUMERICAL_FEATURES),  # Apply numeric transformations to numeric columns
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)  # Apply categorical transformations to categorical columns
    ]
)

# 5. Separate our data into input features and the outcome we want to predict
X = df[FEATURES]  # Input features
y = df[target_variable]  # What we're trying to predict

# 6. Split data into training set (to teach the model) and testing set (to evaluate it)
# This helps us check if our model can generalize to new data it hasn't seen before
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 80% for training, 20% for testing. Random state ensures we get the same split each time (for reproducibility)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# 7. Create a baseline model for comparison
# This is like establishing a benchmark before trying more advanced techniques
basic_pipeline = Pipeline([
    ('preprocessor', preprocessor),  # First process the data
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # Then make predictions
    # n_estimators=100 means we're using 100 decision trees in our "forest"
])

# 8. Train the baseline model and evaluate its performance
# This shows how well our model performs with standard settings
basic_pipeline.fit(X_train, y_train)  # Train the model on our training data
y_pred_basic = basic_pipeline.predict(X_test)  # Make predictions on test data

# Show detailed performance metrics for our baseline model
print("\nBaseline Model Performance:")
print("\nConfusion Matrix:")  # Shows correct and incorrect predictions by category
print(confusion_matrix(y_test, y_pred_basic))
print("\nClassification Report:")  # Shows precision, recall, and F1-score
print(classification_report(y_test, y_pred_basic))

# Try to calculate the AUC score (Area Under Curve) - a measure of how well the model can distinguish between classes
try:
    y_pred_proba_basic = basic_pipeline.predict_proba(X_test)  # Get probability predictions
    auc_basic = roc_auc_score(y_test, y_pred_proba_basic)  # Calculate AUC score
    print(f"ROC-AUC Score: {auc_basic:.4f}")  # Higher is better (max is 1.0)
except:
    print("Could not calculate ROC-AUC score for baseline model")

# 9. Create another pipeline for advanced tuning
# This will help us find the best model settings automatically
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Same preprocessing steps
    ('classifier', RandomForestClassifier(random_state=42))  # But we'll tune this model's settings
])

# 10. Set up automatic hyperparameter tuning
# This is like experimenting with different settings to find the best combination
param_grid = {
    'classifier__n_estimators': [80, 100, 150],  # Try different numbers of trees
    'classifier__max_depth': [None, 20, 30],  # Try different tree depths (None = unlimited)
    'classifier__min_samples_split': [2, 5],  # Minimum samples required to split a node
    'classifier__min_samples_leaf': [1, 2],  # Minimum samples required at each leaf node
    'classifier__max_features': ['sqrt', 'log2']  # How many features to consider at each split
}

# RandomizedSearchCV tries different combinations of the settings above to find the best one
# It's "randomized" because trying all combinations would take too long
random_search = RandomizedSearchCV(
    estimator=rf_pipeline,  # Our pipeline
    param_distributions=param_grid,  # Settings to try
    n_iter=10,  # Try 10 different combinations (instead of all possible ones)
    scoring='accuracy',  # Optimize for accuracy
    cv=5,  # 5-fold cross-validation (split training data into 5 parts for validation)
    verbose=1,  # Show progress
    n_jobs=-1,  # Use all available CPU cores
    random_state=42  # For reproducible results
)

# 11. Run the tuning process (this may take a while)
random_search.fit(X_train, y_train)  # Train models with different settings

# 12. Get the best model found during tuning
best_model = random_search.best_estimator_  # The model with the best performance
print(f"\nBest parameters: {random_search.best_params_}")  # Show the winning settings

# 13. Use the best model to make predictions
y_pred = best_model.predict(X_test)  # Make predictions on test data

# 14. Evaluate how well the optimized model performs
print("\nOptimized Model Performance:")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 15. Find out which features were most important for making predictions
# This helps us understand which factors matter most for our outcome
def feature_importance(model, preproc):
    """Extract and show which features had the biggest impact on predictions."""
    # Get feature names for our report
    feature_names = []
    
    # Add names of numerical features
    num_features = NUMERICAL_FEATURES
    feature_names.extend(num_features)
    
    # Add names of categorical features (expanded after one-hot encoding)
    # For example, if we had "Color" with values "Red" and "Blue", we'd get "Color_Red" and "Color_Blue"
    cat_features = []
    for cat in CATEGORICAL_FEATURES:
        # Get all unique values for this category
        unique_values = df[cat].dropna().unique()
        for val in unique_values:
            cat_features.append(f"{cat}_{val}")
    
    feature_names.extend(cat_features)
    
    # Get importance scores from our model
    importances = model.named_steps['classifier'].feature_importances_
    
    # Sort features by importance (highest first)
    indices = np.argsort(importances)[::-1]
    
    # Print the top features in order of importance
    print("\nFeature Importance Ranking:")
    for i in range(min(20, len(indices))):  # Show top 20 features
        try:
            print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")
        except IndexError:
            print(f"{i+1}. Feature #{indices[i]} ({importances[indices[i]]:.4f})")

# Show feature importance
feature_importance(best_model, preprocessor)