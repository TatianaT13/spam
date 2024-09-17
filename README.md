# Classification de Spams par Email

Ce projet utilise le machine learning pour classifier les emails en spams ou non-spams. Il exploite un modèle Naive Bayes entraîné sur un dataset d'emails.

## Installation

1. Clonez le dépôt :
    ```bash
    git clone https://github.com/TatianaT13/spam.git
    ```
2. Naviguez dans le dossier du projet :
    ```bash
    cd classification_spam
    ```
3. Créez un environnement virtuel et activez-le :
    ```bash
    python -m venv env
    source env/bin/activate  # Sur Windows : env\Scripts\activate
    ```
4. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Données

Les données utilisées dans ce projet proviennent du dataset Enron Email Dataset disponible sur Kaggle.

## Méthodologie

1. **Prétraitement des Textes** : Nettoyage des emails, suppression des stopwords, tokenisation.
2. **Extraction des Caractéristiques** : Utilisation de TF-IDF pour convertir les textes en vecteurs numériques.
3. **Modélisation** : Entraînement d'un modèle Naive Bayes pour la classification.
4. **Évaluation** : Mesure de la performance avec des métriques comme la précision, le rappel et le score F1.

## Résultats

Le modèle atteint une précision élevée et un bon score F1, démontrant son efficacité à identifier les spams.

## Utilisation

Pour exécuter le notebook Jupyter :
```bash
jupyter notebook notebooks/Classification_Spam.ipynb