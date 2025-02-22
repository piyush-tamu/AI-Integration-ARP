# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv('./data.csv')  # Replace with your actual dataset file

# 2. Define numerical and categorical features
NUMERICAL_FEATURES = ['Age']
CATEGORICAL_FEATURES = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level']

# 3. Preprocessing Pipelines
numeric_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 4. Combine Preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, NUMERICAL_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ]
)

# 5. Encode target variable (Disease)
disease_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
df['Disease'] = disease_encoder.fit_transform(df[['Disease']]).argmax(axis=1)  # Convert to numerical labels

# 6. Separate features and target variable
X = df.drop(['Disease', 'Outcome Variable'], axis=1)
y = df['Disease']

# 7. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Create a full pipeline with preprocessing and RandomForestClassifier
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 9. Hyperparameter tuning using RandomizedSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100, 200, 300],
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_grid,
    n_iter=20,  # Number of random combinations to try
    scoring='accuracy',
    cv=5,  # 5-fold cross-validation
    verbose=2,
    n_jobs=-1
)

# 10. Fit the model with best hyperparameters
random_search.fit(X_train, y_train)

# 11. Get the best model
best_model = random_search.best_estimator_

# 12. Make predictions
y_pred = best_model.predict(X_test)

# 13. Evaluate performance
print("\nOptimized Model Performance:")
print(classification_report(y_test, y_pred))

# 14. Function to predict disease for new patients
def predict_disease(model, input_data):
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure the input follows the training column order
    input_df = input_df[X_train.columns]

    # Make a prediction (pipeline will handle preprocessing)
    prediction = model.predict(input_df)  # Returns an encoded class (integer index)

    # Convert numerical prediction back to disease name
    # Since prediction is now a single index, we find the original class name
    # we have to access the categories in the encoder to find the name of the class
    
    # Get the category names, which was the original labels of the 'Disease' column
    disease_categories = disease_encoder.categories_[0]
    
    # Use the index from prediction to get corresponding category name
    disease_name = disease_categories[prediction[0]]
    
    return disease_name



# Example prediction (Replace with actual patient data)
sample_input = {
    'Fever': 'Yes', 'Cough': 'No', 'Fatigue': 'Yes', 'Difficulty Breathing': 'No',
    'Age': 45, 'Gender': 'Male', 'Blood Pressure': 'Normal', 'Cholesterol Level': 'High'
}

predicted_disease = predict_disease(best_model, sample_input)
print("\nPredicted Disease for Sample Input:", predicted_disease)