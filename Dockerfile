# Utilisation de python:3.11-slim pour une base légère
FROM python:3.11-slim

# Définition du répertoire de travail
WORKDIR /app

# Installation des dépendances système nécessaires en une seule commande RUN et nettoyage après installation
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
# Copie du fichier requirements.txt uniquement
COPY requirements.txt /app/requirements.txt

# Installation des dépendances Python et téléchargement des modèles spaCy en une seule couche RUN
RUN pip3 install --no-cache-dir --default-timeout=1000 -r requirements.txt && \
    python3 -m spacy download fr_core_news_sm && \
    python3 -m spacy download en_core_web_sm && \
    python3 -m spacy download de_core_news_sm && \
    python3 -m spacy download es_core_news_sm && \
    python3 -m spacy download it_core_news_sm && \
    python3 -m spacy download nl_core_news_sm && \
    python3 -m spacy download ca_core_news_sm && \
    python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"


# Copie de tous les autres fichiers nécessaires
COPY . /app/

# Exposition du port utilisé par l'application
EXPOSE 8501

# Vérification de la santé de l'application
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Point d'entrée pour démarrer l'application
ENTRYPOINT ["streamlit", "run", "RAKUTEN.py", "--server.port=8501", "--server.address=0.0.0.0",  "--server.enableCORS=false", "--server.headless=true", "--server.enableXsrfProtection=false","--server.baseUrlPath=/rakuten"]
