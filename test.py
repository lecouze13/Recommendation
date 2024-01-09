import spacy
from spacy.training.example import Example
import pdfplumber
from spacy.training import offsets_to_biluo_tags

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text(encoding='utf-8')  # Spécifie l'encodage
    return text

# Charger le modèle SpaCy pour la langue française ou un modèle pré-entraîné
nlp_telephone = spacy.blank("fr")

# Créer un entity recognizer pour "TELEPHONE"
ner = nlp_telephone.add_pipe("ner")
ner.add_label("TELEPHONE")

# Données annotées pour "TELEPHONE" avec des positions réalignées
label_data_telephone = [
    {"text": "1234567890.", "entities": [(0, 10, "TELEPHONE")]},
    {"text": "0687601273", "entities": [(0, 10, "TELEPHONE")]},
    {"text": "tel:0687601273", "entities": [(4, 14, "TELEPHONE")]},
    {"text": "0687601273", "entities": [(0, 10, "TELEPHONE")]}
    # Ajoutez d'autres exemples annotés pour "TELEPHONE" ici
]

# Vérifier l'alignement des entités annotées avec offsets_to_biluo_tags
for entry in label_data_telephone:
    text = entry["text"]
    entities = entry["entities"]
    doc = nlp_telephone.make_doc(text)
    biluo_tags = offsets_to_biluo_tags(doc, entities)

# Créer les exemples pour l'entraînement du modèle
examples = []
for entry in label_data_telephone:
    text = entry["text"]
    entities = entry["entities"]
    example = Example.from_dict(nlp_telephone.make_doc(text), {"entities": entities})
    examples.append(example)

# Entraîner le modèle avec les exemples
nlp_telephone.initialize()
optimizer = nlp_telephone.begin_training()
for iteration in range(10):  # Nombre d'itérations pour l'entraînement
    losses = {}
    nlp_telephone.update(examples, drop=0.5, losses=losses)

# Chemin vers votre CV au format PDF
pdf_path = "CV-14.pdf"
cv_text = extract_text_from_pdf(pdf_path)

# Traiter le texte extrait avec le modèle SpaCy entraîné
doc_cv = nlp_telephone(cv_text)

# Afficher les numéros de téléphone trouvés dans le CV
for ent in doc_cv.ents:
    if ent.label_ == "TELEPHONE":
        print(ent.text)
