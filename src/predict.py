import joblib
import pandas as pd

def clean_text(text):
    """
    Nettoie le texte en retirant la ponctuation, 
    convertissant en minuscules, et en supprimant les stopwords.
    """
    import string
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def load_models():
    """
    Charge le modèle Naive Bayes et le vectoriseur TF-IDF à partir des fichiers sauvegardés.
    """
    model = joblib.load('spam_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

def predict_message(model, vectorizer, message):
    """
    Prend un message en entrée, le nettoie, le transforme avec le vectoriseur TF-IDF,
    puis utilise le modèle pour prédire si c'est un spam ou non.
    """
    # Nettoyer le message
    cleaned_message = clean_text(message)
    
    # Transformer le message en vecteur TF-IDF
    message_vector = vectorizer.transform([cleaned_message]).toarray()
    
    # Prédire avec le modèle
    prediction = model.predict(message_vector)
    
    # Afficher le résultat
    if prediction[0] == 1:
        print("Ce message est un SPAM.")
    else:
        print("Ce message n'est PAS un spam.")

def main():
    # Charger les modèles
    model, vectorizer = load_models()
    
    # Demander à l'utilisateur d'entrer un message
    message = input("Entrez un message SMS pour prédire s'il est un spam : ")
    
    # Prédire si le message est un spam
    predict_message(model, vectorizer, message)

if __name__ == "__main__":
    main()