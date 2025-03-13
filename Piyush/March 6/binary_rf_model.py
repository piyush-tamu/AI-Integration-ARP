# import necessary libraries
import string
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# input observation and symptom detection

# bovine tuberculosis
observation = "The animal was brought into the vet clinic exhibiting lethargy, swelling, and a fever. There was also significant weight loss." 
observations = observation.split()

# possible symptoms
symptoms_list = [
    "sudden_death", "blood_from_nose", "trembling", "difficult_breathing", "blood_from_openings", "fever", "loss_of_appetite", "dullness", "swelling",
    "recumnency", "profuse_salivation", "vesicles", "lameness", "change_in_behaviour", "furious", "dumbness", "nasal_discharge", "eye_discharge", 
    "haemorrages", "lethargy", "enteritis", "abortion", "no_breed", "unwillingness", "stiffness", "eraction", "mastication", "paralysis", "encephalitis", 
    "septicaemia", "infertility", "nacrotic_foci", "diarrhea", "weight_loss", "shivering", "drooling", "excessive_urination"
]

symptom_mapping = {}

# Create versions with spaces instead of underscores for detection
for symptom in symptoms_list:
    natural_form = symptom.replace('_', ' ')
    symptom_mapping[natural_form] = symptom

cleaned_observation = observation.lower()
# Remove punctuation but maintain spaces
for punct in string.punctuation:
    cleaned_observation = cleaned_observation.replace(punct, '')

# Detect symptoms from the observation
symptoms = []
for natural_symptom, coded_symptom in symptom_mapping.items():
    if natural_symptom in cleaned_observation:
        symptoms.append(coded_symptom)

# Create feature vector
feature_vector = [1 if symptom in symptoms else 0 for symptom in symptoms_list]

print(f"Input observation: {observation}")
print(f"Detected symptoms: {symptoms}")
print(f"Feature vector: {feature_vector}")

# Load and prepare dataset
df = pd.read_csv('./animal.csv')
print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")

# Split data into train and test sets
train_data = []
test_data = []

for i, row in df.iterrows():
    if (i % 5) < 4:  # 80% training data
        train_data.append(row)
    else:            # 20% testing data
        test_data.append(row)

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

print(f"Training set size: {len(train_df)} samples")
print(f"Testing set size: {len(test_df)} samples")

# Prepare features and target variables
FEATURES = [col for col in df.columns if col != 'prognosis']
print(f"\nTotal number of features used: {len(FEATURES)}")

X_train = train_df[FEATURES]
y_train = train_df['prognosis']
X_test = test_df[FEATURES]
y_test = test_df['prognosis']

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model training completed!")

# Model evaluation
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Make prediction for new observation
observation_df = pd.DataFrame([feature_vector], columns=FEATURES)
observation_prediction = model.predict(observation_df)
print(f"Predicted disease: {observation_prediction[0]}")

print("")

# returns the probability of each class for each observation
probabilities = model.predict_proba(observation_df)[0]
n_top_predictions = 3
top_indices = probabilities.argsort()[-n_top_predictions:][::-1]
predicted_diseases = [model.classes_[index] for index in top_indices]
print("Top 3 Potential Diseases:")
for disease, prob in zip(predicted_diseases, probabilities[top_indices]):
    print(f"- {disease}: {prob:.4f}") 