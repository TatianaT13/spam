import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import string
import joblib

# Télécharger les stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Nettoie le texte en retirant la ponctuation, 
    convertissant en minuscules, et en supprimant les stopwords.
    """
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def load_and_prepare_data(path):
    """
    Charge le dataset et effectue le nettoyage nécessaire.
    Renvoie un DataFrame avec les colonnes 'label' et 'text'.
    """
    # Charger seulement les colonnes nécessaires ('v1' et 'v2')
    data = pd.read_csv(path, usecols=['v1', 'v2'], encoding='latin1')
    data.columns = ['label', 'text']  # Renommer les colonnes pour plus de clarté
    data['label'] = data['label'].map({'spam': 1, 'ham': 0})  # Convertir les labels en numériques
    data['text'] = data['text'].apply(clean_text)  # Appliquer le nettoyage
    return data

def train_model(X, y):
    """
    Entraîne un modèle Naive Bayes Multinomial sur les données fournies.
    Renvoie le modèle entraîné.
    """
    model = MultinomialNB()
    model.fit(X, y)
    return model

def main():
    # Charger et préparer les données
    data = load_and_prepare_data('data/spam.csv')
    print("Aperçu des données après nettoyage :")
    print(data.head())
    
    # Extraction des caractéristiques avec TF-IDF
    tfidf = TfidfVectorizer(max_features=3000)  # Limiter à 3000 caractéristiques
    X = tfidf.fit_transform(data['text']).toarray()
    y = data['label']
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraîner le modèle
    model = train_model(X_train, y_train)
    
    # Évaluer le modèle
    y_pred = model.predict(X_test)
    print("\nRapport de Classification :")
    print(classification_report(y_test, y_pred))
    
    # Sauvegarder le modèle et le vectoriseur
    joblib.dump(model, 'spam_classifier_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    print("Modèle et vectoriseur sauvegardés")

if __name__ == "__main__":
    main()