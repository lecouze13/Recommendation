import json
import re
import spacy
import pdfplumber
from spacy.training.example import Example

# Charger le modèle SpaCy avec les composants NER (Named Entity Recognition) pré-entraînés
nlp = spacy.load("fr_core_news_sm")

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text(encoding='utf-8')
    return text

def extract_phone_numbers(text):
    phone_numbers = re.findall(r'\b\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}\b', text)
    return phone_numbers

def extract_emails(text):
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return emails

def extract_languages_with_level(text):
    languages = re.findall(r'\b(?:français|anglais|espagnol|allemand)\b(?:\s(?:A|B|C)\d)?', text, flags=re.IGNORECASE)
    return languages

def extract_addresses(text):
    addresses = re.findall(r'\b\d{1,4}\s\w+\s\w+\b', text)
    return addresses

def load_training_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        training_data = json.load(file)
    return training_data

# Lecture des données annotées pour l'entraînement du modèle
training_data = load_training_data('train_data2.json')

# Entraînement du modèle SpaCy
nlp = spacy.blank("fr")
ner = nlp.add_pipe("ner", last=True)

# Ajout des labels
ner.add_label("PER")
ner.add_label("EXPERIENCE")
ner.add_label("LANG_LEVEL")
# Ajoute d'autres labels nécessaires (PHONE, EMAIL, etc.)

# Création des exemples pour l'entraînement du modèle
examples = []
for data in training_data:
    text = data["text"]
    entities = []
    for entity in data["entities"]:
        start, end, label = entity["start"], entity["end"], entity["label"]
        entities.append((start, end, label))
    examples.append(Example.from_dict(nlp.make_doc(text), {"entities": entities}))

# Entraînement du modèle
nlp.initialize(lambda: examples)
for i in range(10):  # Nombre d'itérations d'entraînement
    losses = {}
    nlp.update(examples, drop=0.5, losses=losses)

# Sauvegarde du modèle entraîné
nlp.to_disk("modele_cv")

# Chargement du modèle SpaCy entraîné
nlp_loaded = spacy.load("modele_cv")

# Extraction des informations du nouveau CV
pdf_path = "CV-14.pdf"
extracted_text = extract_text_from_pdf(pdf_path)

# Analyse du texte extrait avec le modèle chargé
doc = nlp_loaded(extracted_text)

# Extraction des entités nommées du CV
for ent in doc.ents:
    print(ent.text, ent.label_)

# Extraction des numéros de téléphone, des adresses e-mail, des langues et des adresses du texte brut
phone_numbers = extract_phone_numbers(extracted_text)
emails = extract_emails(extracted_text)
languages = extract_languages_with_level(extracted_text)
addresses = extract_addresses(extracted_text)

print("Numéros de téléphone trouvés :")
print(phone_numbers)
print("\nAdresses e-mail trouvées :")
print(emails)
print("\nLangues détectées :")
print(languages)
print("\nAdresses trouvées :")
print(addresses)
