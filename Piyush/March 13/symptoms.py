import csv
import random

def generate_disease_data(output_file="disease_data.csv"):

    columns = [
        "sudden_death", "blood_from_nose", "trembling", "difficult_breathing",
        "blood_from_openings", "fever", "loss_of_appetite", "dullness",
        "swelling", "recumbency", "profuse_salivation", "vesicles", "lameness",
        "change_in_behaviour", "furious", "dumbness", "nasal_discharge",
        "eye_discharge", "hemorrhages", "lethargy", "enteritis", "abortion",
        "no_breed", "unwillingness", "stiffness", "reaction_mastication",
        "paralysis", "encephalitis", "septicemia", "infertility",
        "necrotic_foci", "diarrhea", "weight_loss", "shivering", "drooling",
        "excessive_urination", "coughing", "hair_loss", "constipation",
        "tachycardia", "convulsions/seizures", "tachypnea", "vomiting",
        "scratching", "skin_discoloration", "crepitation", "lesions", "ataxia",
        "enlarged_lymph_nodes", "prognosis"
    ]

    diseases = [
        "Pyoderma", "Gastric Dilatation-Volvulus (GDV)", "Chronic Kidney Disease",
        "Liver Disease", "Cataracts", "Glaucoma", "Cherry Eye", "Dental Disease",
        "Obesity", "Cancer (Various Types)", "Hemangiosarcoma", "Osteosarcoma",
        "Lymphoma", "Mast Cell Tumors", "Bladder Stones",
        "Urinary Tract Infection", "Pyometra", "Prostatic Disease",
        "Testicular Tumors", "Anal Sac Disease", "Inflammatory Bowel Disease",
        "Colitis", "Gastroenteritis", "Food Allergies", "Flea Allergy Dermatitis",
        "Tick-Borne Diseases", "Fungal Infections", "Bacterial Infections",
        "Viral Infections", "Heatstroke", "Frostbite", "Poisoning (Toxicities)",
        "Chocolate Toxicity", "Xylitol Toxicity", "Rat Poison Toxicity",
        "Snake Bites", "Bee Stings", "Autoimmune Diseases", "Lupus",
        "Pemphigus", "Immune-Mediated Hemolytic Anemia",
        "Immune-Mediated Thrombocytopenia", "Congestive Heart Failure",
        "Dilated Cardiomyopathy", "Mitral Valve Disease", "Heart Murmurs",
        "Pulmonary Hypertension", "Tracheal Collapse", "Brachycephalic Syndrome",
        "Pneumonia", "Chronic Bronchitis", "Asthma", "Nasal Tumors", "Sinusitis",
        "Oral Tumors", "Esophageal Disease", "Megaesophagus"
    ]

    disease_descriptions = {
        "Pyoderma": "Pustules, redness, itching, and hair loss, often secondary to allergies or infections.",
        "Gastric Dilatation-Volvulus (GDV)": "Distended abdomen, unproductive vomiting, restlessness, rapid breathing, and collapse.",
        "Chronic Kidney Disease": "Increased thirst, frequent urination, weight loss, vomiting, lethargy, and bad breath.",
        "Liver Disease": "Jaundice, vomiting, diarrhea, lethargy, weight loss, and abdominal swelling.",
        "Cataracts": "Cloudy or opaque appearance of the eyes, vision loss, and clumsiness.",
        "Glaucoma": "Redness, pain, cloudiness of the eye, vision loss, and increased eye pressure.",
        "Cherry Eye": "Prolapse of the third eyelid gland, causing a red mass in the corner of the eye.",
        "Dental Disease": "Bad breath, drooling, difficulty eating, tooth loss, and gum inflammation.",
        "Obesity": "Excessive weight, difficulty exercising, lethargy, and increased risk of other diseases.",
        "Cancer (Various Types)": "Lumps, weight loss, lethargy, loss of appetite, bleeding, and organ-specific symptoms (e.g., lameness with bone cancer).",
        "Hemangiosarcoma": "Weakness, collapse, pale gums, abdominal swelling, and sudden death.",
        "Osteosarcoma": "Lameness, swelling at the site of the tumor, pain, and fractures.",
        "Lymphoma": "Swollen lymph nodes, lethargy, weight loss, vomiting, and diarrhea.",
        "Mast Cell Tumors": "Skin lumps, redness, itching, and gastrointestinal ulcers in severe cases.",
        "Bladder Stones": "Straining to urinate, blood in urine, frequent urination, and pain.",
        "Urinary Tract Infection": "Frequent urination, straining to urinate, blood in urine, and licking of the genital area.",
        "Pyometra": "Lethargy, vomiting, increased thirst, vaginal discharge (in open pyometra), and abdominal swelling.",
        "Prostatic Disease": "Difficulty urinating, blood in urine, straining to defecate, and lethargy.",
        "Testicular Tumors": "Swelling of the testicles, abdominal swelling, and hormonal changes (e.g., feminization in male dogs).",
        "Anal Sac Disease": "Scooting, licking or biting at the rear, foul odor, and swelling near the anus.",
        "Inflammatory Bowel Disease": "Chronic vomiting, diarrhea, weight loss, and lethargy.",
        "Colitis": "Diarrhea, often with mucus or blood, straining to defecate, and urgency.",
        "Gastroenteritis": "Vomiting, diarrhea, lethargy, and abdominal pain.",
        "Food Allergies": "Itching, skin infections, ear infections, and gastrointestinal upset.",
        "Flea Allergy Dermatitis": "Intense itching, hair loss, redness, and skin infections, often around the tail base and hindquarters.",
        "Tick-Borne Diseases": "Fever, lethargy, joint pain, lameness, and anemia (e.g., Lyme, Ehrlichiosis, Anaplasmosis).",
        "Fungal Infections": "Skin lesions, hair loss, coughing, and systemic signs (e.g., fever, lethargy) in systemic fungal infections.",
        "Bacterial Infections": "Fever, lethargy, localized swelling, redness, and discharge (depending on the site of infection).",
        "Viral Infections": "Fever, lethargy, coughing, nasal discharge, and gastrointestinal signs (depending on the virus).",
        "Heatstroke": "Excessive panting, drooling, vomiting, diarrhea, collapse, and seizures.",
        "Frostbite": "Pale or blue skin, swelling, pain, and tissue necrosis in severe cases.",
        "Poisoning (Toxicities)": "Vomiting, diarrhea, seizures, lethargy, and organ failure (depending on the toxin).",
        "Chocolate Toxicity": "Vomiting, diarrhea, increased heart rate, seizures, and death in severe cases.",
        "Xylitol Toxicity": "Vomiting, weakness, collapse, seizures, and liver failure.",
        "Rat Poison Toxicity": "Bleeding, lethargy, pale gums, coughing up blood, and blood in stool or urine.",
        "Snake Bites": "Swelling, pain, bruising, lethargy, and systemic signs (e.g., shock, organ failure) in severe cases.",
        "Bee Stings": "Swelling, pain, redness, and allergic reactions (e.g., facial swelling, difficulty breathing) in severe cases.",
        "Autoimmune Diseases": "Varies by disease (e.g., skin lesions, joint pain, anemia, and organ-specific signs).",
        "Lupus": "Skin lesions, joint pain, fever, and organ-specific signs (e.g., kidney failure).",
        "Pemphigus": "Blisters, crusting, and skin erosions, often on the face and paws.",
        "Immune-Mediated Hemolytic Anemia": "Pale gums, lethargy, weakness, jaundice, and dark urine.",
        "Immune-Mediated Thrombocytopenia": "Bruising, bleeding, lethargy, and pale gums.",
        "Congestive Heart Failure": "Coughing, difficulty breathing, lethargy, and abdominal swelling.",
        "Dilated Cardiomyopathy": "Weakness, coughing, difficulty breathing, and collapse.",
        "Mitral Valve Disease": "Coughing, difficulty breathing, lethargy, and exercise intolerance.",
        "Heart Murmurs": "Often asymptomatic, but may cause lethargy, coughing, or exercise intolerance in severe cases.",
        "Pulmonary Hypertension": "Difficulty breathing, coughing, lethargy, and collapse.",
        "Tracheal Collapse": "Honking cough, difficulty breathing, and exercise intolerance.",
        "Brachycephalic Syndrome": "Noisy breathing, snoring, exercise intolerance, and difficulty breathing.",
        "Pneumonia": "Coughing, difficulty breathing, fever, lethargy, and nasal discharge.",
        "Chronic Bronchitis": "Persistent cough, difficulty breathing, and exercise intolerance.",
        "Asthma": "Coughing, wheezing, difficulty breathing, and exercise intolerance.",
        "Nasal Tumors": "Nasal discharge, nosebleeds, sneezing, and facial swelling.",
        "Sinusitis": "Nasal discharge, sneezing, and facial pain.",
        "Oral Tumors": "Drooling, difficulty eating, bad breath, and swelling in the mouth.",
        "Esophageal Disease": "Regurgitation, difficulty swallowing, and weight loss.",
        "Megaesophagus": "Regurgitation, weight loss, and aspiration pneumonia."
    }

    data = []
    for disease in diseases:
        for _ in range(50):
            row = {}
            for col in columns:
                if col == "prognosis":
                    row[col] = disease
                else:
                    row[col] = 0  # Default to 0 (absence)

            # Simulate symptom presence based on disease descriptions
            description = disease_descriptions[disease].lower()
            if "sudden death" in description:
                if random.random() < 0.1: #small chance of sudden death
                    row["sudden_death"] = 1
            if "blood from nose" in description:
                if random.random() < 0.05:
                    row["blood_from_nose"] = 1
            if "trembling" in description:
                if random.random() < 0.1:
                    row["trembling"] = 1
            if "difficult breathing" in description or "difficulty breathing" in description:
                if random.random() < 0.5:
                    row["difficult_breathing"] = 1
            if "blood from openings" in description:
                if random.random() < 0.05:
                    row["blood_from_openings"] = 1
            if "fever" in description:
                if random.random() < 0.4:
                    row["fever"] = 1
            if "loss of appetite" in description:
                if random.random() < 0.4:
                    row["loss_of_appetite"] = 1
            if "lethargy" in description or "dullness" in description:
                if random.random() < 0.6:
                    row["lethargy"] = 1
                    row["dullness"] = 1
            if "swelling" in description:
                if random.random() < 0.4:
                    row["swelling"] = 1
            if "recumbency" in description:
                if random.random() < 0.2:
                    row["recumbency"] = 1
            if "profuse salivation" in description or "drooling" in description:
                if random.random() < 0.3:
                    row["profuse_salivation"] = 1
                    row["drooling"] = 1
            if "vesicles" in description:
                if random.random() < 0.1:
                    row["vesicles"] = 1
            if "lameness" in description:
                if random.random() < 0.3:
                    row["lameness"] = 1
            if "change in behaviour" in description:
                if random.random() < 0.2:
                    row["change_in_behaviour"] = 1
            if "furious" in description:
                if random.random() < 0.05:
                    row["furious"] = 1
            if "dumbness" in description:
                if random.random() < 0.05:
                    row["dumbness"] = 1
            if "nasal discharge" in description:
                if random.random() < 0.3:
                    row["nasal_discharge"] = 1
            if "eye discharge" in description:
                if random.random() < 0.2:
                    row["eye_discharge"] = 1
            if "hemorrhages" in description:
                if random.random() < 0.1:
                    row["hemorrhages"] = 1
            if "enteritis" in description:
                if random.random() < 0.2:
                    row["enteritis"] = 1
            if "abortion" in description:
                if random.random() < 0.05:
                    row["abortion"] = 1
            if "no breed" in description:
                if random.random() < 0.05:
                    row["no_breed"] = 1
            if "unwillingness" in description:
                if random.random() < 0.1:
                    row["unwillingness"] = 1
            if "stiffness" in description:
                if random.random() < 0.2:
                    row["stiffness"] = 1
            if "reaction mastication" in description:
                if random.random() < 0.05:
                    row["reaction_mastication"] = 1
            if "paralysis" in description:
                if random.random() < 0.1:
                    row["paralysis"] = 1
            if "encephalitis" in description:
                if random.random() < 0.05:
                    row["encephalitis"] = 1
            if "septicemia" in description:
                if random.random() < 0.05:
                    row["septicemia"] = 1
            if "infertility" in description:
                if random.random() < 0.05:
                    row["infertility"] = 1
            if "necrotic foci" in description:
                if random.random() < 0.05:
                    row["necrotic_foci"] = 1
            if "diarrhea" in description:
                if random.random() < 0.5:
                    row["diarrhea"] = 1
            if "weight loss" in description:
                if random.random() < 0.4:
                    row["weight_loss"] = 1
            if "shivering" in description:
                if random.random() < 0.1:
                    row["shivering"] = 1
            if "excessive urination" in description or "frequent urination" in description:
                if random.random() < 0.4:
                    row["excessive_urination"] = 1
            if "coughing" in description:
                if random.random() < 0.4:
                    row["coughing"] = 1
            if "hair loss" in description:
                if random.random() < 0.3:
                    row["hair_loss"] = 1
            if "constipation" in description:
                if random.random() < 0.1:
                    row["constipation"] = 1
            if "tachycardia" in description or "increased heart rate" in description:
                if random.random() < 0.2:
                    row["tachycardia"] = 1
            if "convulsions" in description or "seizures" in description:
                if random.random() < 0.3:
                    row["convulsions/seizures"] = 1
            if "tachypnea" in description or "rapid breathing" in description:
                if random.random() < 0.3:
                    row["tachypnea"] = 1
            if "vomiting" in description:
                if random.random() < 0.5:
                    row["vomiting"] = 1
            if "scratching" in description or "itching" in description:
                if random.random() < 0.4:
                    row["scratching"] = 1
            if "skin discoloration" in description or "redness" in description or "pale gums" in description or "jaundice" in description:
                if random.random() < 0.4:
                    row["skin_discoloration"] = 1
            if "crepitation" in description:
                if random.random() < 0.05:
                    row["crepitation"] = 1
            if "lesions" in description:
                if random.random() < 0.3:
                    row["lesions"] = 1
            if "ataxia" in description or "clumsiness" in description:
                if random.random() < 0.2:
                    row["ataxia"] = 1
            if "enlarged lymph nodes" in description or "swollen lymph nodes" in description:
                if random.random() < 0.3:
                    row["enlarged_lymph_nodes"] = 1

            data.append(row)

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print(f"CSV file '{output_file}' generated successfully.")

if __name__ == "__main__":
    generate_disease_data()