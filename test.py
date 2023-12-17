import pdfplumber
import spacy

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text(encoding='utf-8')  # Spécifie l'encodage
    return text

def extract_information(cv_text):
    nlp = spacy.load("fr_core_news_sm")
    doc = nlp(cv_text)
    
    name = ""
    phone_number = ""
    experience = ""
    email = ""

    for entity in doc.ents:
        if entity.label_ == "PER":  # Nom de personne
            name = entity.text
        elif entity.label_ == "PHONE":  # Numéro de téléphone
            phone_number = entity.text
        elif entity.label_ == "EXP":  # Expérience
            experience = entity.text
        elif entity.label_ == "EMAIL":  # Email
            email = entity.text
    
    return {
        "Nom": name,
        "Numéro de téléphone": phone_number,
        "Expérience": experience,
        "Email": email
    }

pdf_path = "../RECOMMENDATION/Recommendation/CV.pdf"
cv_text = extract_text_from_pdf(pdf_path)
cv_info = extract_information(cv_text)

print("Informations extraites du CV:")
print(cv_info)
