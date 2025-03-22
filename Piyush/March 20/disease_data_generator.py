import json
import pandas as pd
import numpy as np
import random
import os

def load_disease_data(json_file_path):
    """Load disease data from a JSON file."""
    with open(json_file_path, 'r') as file:
        return json.load(file)

def load_symptom_list(json_file_path):
    """Load the comprehensive list of symptoms."""
    with open(json_file_path, 'r') as file:
        return json.load(file)

def generate_synthetic_data(disease_data, symptom_list, rows_per_disease=50):
    # Use a list instead of a set to preserve order
    all_symptoms = symptom_list['symptoms']
    
    # Create empty DataFrame with defined column order
    columns = all_symptoms + ['disease']
    synthetic_data = pd.DataFrame(columns=columns)
    
    # Process each disease
    for disease_info in disease_data:
        disease_name = disease_info['disease']
        symptom_probs = disease_info['symptoms']
        
        # Create rows for this disease
        disease_rows = []
        
        for _ in range(rows_per_disease):
            # Start with all symptoms set to 0
            row_data = {symptom: 0 for symptom in all_symptoms}
            
            # Set disease column
            row_data['disease'] = disease_name
            
            # Set symptoms based on their probabilities
            for symptom, prob_str in symptom_probs.items():
                # Convert percentage string to float probability
                prob = float(prob_str.strip('%')) / 100
                prob = min(prob, 1.0)
                
                # Randomly determine if this symptom is present
                if random.random() < prob:
                    row_data[symptom] = 1
            
            disease_rows.append(row_data)
        
        # Add this disease's rows to the main DataFrame
        disease_df = pd.DataFrame(disease_rows)
        # Ensure columns are in the right order
        disease_df = disease_df[columns]
        synthetic_data = pd.concat([synthetic_data, disease_df], ignore_index=True)
    
    return synthetic_data

def save_and_append_data(synthetic_data, output_file, original_file=None, append=False):
    """
    Save synthetic data to a CSV file and optionally append to original file.
    
    Args:
        synthetic_data: DataFrame containing the synthetic data
        output_file: Path to save the synthetic data
        original_file: Path to the original CSV file (optional)
        append: Whether to append synthetic data to original file
    """
    # Save synthetic data to CSV
    synthetic_data.to_csv(output_file, index=False)
    print(f"Synthetic data saved to {output_file}")
    
    # If append option is selected and original file exists
    if append and original_file and os.path.exists(original_file):
        original_data = pd.read_csv(original_file)
        
        # Check if column headers match
        if set(original_data.columns) == set(synthetic_data.columns):
            combined_data = pd.concat([original_data, synthetic_data], ignore_index=True)
            combined_data.to_csv(original_file, index=False)
            print(f"Synthetic data appended to {original_file}")
        else:
            print("Column headers don't match. Could not append to original file.")

def main():
    # File paths
    disease_data_file = "paste.txt"  # Your first JSON file
    symptom_list_file = "paste-2.txt"  # Your second JSON file
    output_file = "synthetic_disease_data.csv"  # Output file for synthetic data
    
    # Original CSV file path (if you want to append)
    original_csv = None  # Set this to your original CSV path if you want to append
    
    # Parameters
    rows_per_disease = 50  # Number of rows to generate per disease
    append_to_original = False  # Whether to append to original file
    
    # Load data
    disease_data = load_disease_data(disease_data_file)
    symptom_list = load_symptom_list(symptom_list_file)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(disease_data, symptom_list, rows_per_disease)
    
    # Save (and optionally append) the synthetic data
    save_and_append_data(synthetic_data, output_file, original_csv, append_to_original)
    
    # Print some statistics
    print(f"Generated {len(synthetic_data)} rows of synthetic data")
    print(f"Number of columns: {len(synthetic_data.columns)}")
    print(f"Diseases included: {synthetic_data['disease'].nunique()}")

if __name__ == "__main__":
    main()