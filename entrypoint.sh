#!/bin/bash
# =============================================================================
# entrypoint.sh
# Script de démarrage Docker : télécharge les fichiers depuis S3 puis lance
# l'application Streamlit.
#
# Variables d'environnement requises :
#   S3_BUCKET       : nom du bucket S3     (ex: mon-bucket-rakuten)
#   S3_PREFIX       : préfixe/dossier S3   (ex: input/ ou laissez vide)
#   AWS_ACCESS_KEY_ID     : clé d'accès AWS (optionnel si rôle IAM)
#   AWS_SECRET_ACCESS_KEY : clé secrète AWS (optionnel si rôle IAM)
#   AWS_DEFAULT_REGION    : région AWS      (ex: eu-west-1)
# =============================================================================

set -e

INPUT_DIR="/app/input"
FICHIERS_DIR="/app/fichiers"
mkdir -p "$INPUT_DIR"
mkdir -p "$FICHIERS_DIR"

# ---------------------------------------------------------------------------
# Liste des fichiers à télécharger depuis S3 vers /app/input
# Ajoutez ici tous les fichiers nécessaires (un par ligne)
# ---------------------------------------------------------------------------
FILES_TO_DOWNLOAD=(
    "X_train_update.csv"
    "EfficientNetB1_weight.h5"
    "Dfcontour.csv"
    "DfColorMean.json"
)

# ---------------------------------------------------------------------------
# Liste des fichiers à télécharger depuis S3 vers /app/fichiers
# ---------------------------------------------------------------------------
FICHIERS_TO_DOWNLOAD=(
    "RandomForestClassifier_dump.joblib"
    "GradientBoosting_dump.joblib"
    "EfficientNetB1_CONCAT2_X_train.pkl"
    "EfficientNetB1_CONCAT2_X_test.pkl"
    "EMBEDDING_GRU_CONCAT2_X_train.pkl"
    "EMBEDDING_CONCAT2_X_train.pkl"
    "LinearSVC_CONCAT2_X_train.pkl"
    "LinearSVC_CONCAT2_X_test.pkl"
)

# ---------------------------------------------------------------------------
# Vérification de la configuration S3
# ---------------------------------------------------------------------------
if [ -z "$S3_BUCKET" ]; then
    echo "[WARN] La variable S3_BUCKET n'est pas définie. Le téléchargement S3 est ignoré."
else
    echo "[INFO] Bucket S3 : s3://${S3_BUCKET}/${S3_PREFIX}"

    # Vérification de la présence d'awscli
    if ! command -v aws &> /dev/null; then
        echo "[ERROR] aws CLI introuvable. Vérifiez que awscli est installé dans le Dockerfile."
        exit 1
    fi

    for FILE in "${FILES_TO_DOWNLOAD[@]}"; do
        DEST="$INPUT_DIR/$FILE"
        S3_URI="s3://${S3_BUCKET}/${S3_PREFIX}${FILE}"

        # Ne télécharge que si le fichier est absent (évite de re-télécharger à chaque restart)
        if [ -f "$DEST" ]; then
            echo "[SKIP] $FILE déjà présent dans $INPUT_DIR"
        else
            echo "[INFO] Téléchargement de $S3_URI → $DEST ..."
            if aws s3 cp "$S3_URI" "$DEST"; then
                echo "[OK]   $FILE téléchargé avec succès."
            else
                echo "[ERROR] Échec du téléchargement de $S3_URI"
                echo "        Vérifiez les droits, le nom du bucket et le préfixe."
                exit 1
            fi
        fi
    done

    for FILE in "${FICHIERS_TO_DOWNLOAD[@]}"; do
        DEST="$FICHIERS_DIR/$FILE"
        S3_URI="s3://${S3_BUCKET}/${S3_PREFIX}${FILE}"

        if [ -f "$DEST" ]; then
            echo "[SKIP] $FILE déjà présent dans $FICHIERS_DIR"
        else
            echo "[INFO] Téléchargement de $S3_URI → $DEST ..."
            if aws s3 cp "$S3_URI" "$DEST"; then
                echo "[OK]   $FILE téléchargé avec succès."
            else
                echo "[ERROR] Échec du téléchargement de $S3_URI"
                echo "        Vérifiez les droits, le nom du bucket et le préfixe."
                exit 1
            fi
        fi
    done

    echo "[INFO] Tous les fichiers S3 sont prêts."
fi

# ---------------------------------------------------------------------------
# Démarrage de l'application Streamlit
# ---------------------------------------------------------------------------
echo "[INFO] Démarrage de Streamlit..."
exec streamlit run RAKUTEN.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.headless=true \
    --server.enableXsrfProtection=false \
    --server.baseUrlPath=/rakuten
