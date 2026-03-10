# Análisis de Sentimiento en Tweets — TF-IDF vs RoBERTa

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![transformers](https://img.shields.io/badge/transformers-4.30%2B-orange)](https://huggingface.co/docs/transformers/index)

Proyecto de **clasificación de sentimiento** de tweets en inglés, comparando enfoques clásicos (TF-IDF + modelos de ML) versus un transformer moderno pre-entrenado en tweets (`twitter-roberta-base-sentiment-latest`).

**Objetivo principal**:  
Determinar si modelos clásicos bien calibrados siguen siendo competitivos (o superiores) frente a transformers en datasets antiguos (~2009–2011), y entender el impacto del *domain shift* temporal.

### Resultados destacados

- **Mejor modelo clásico** (LogisticRegression / RandomForest con TF-IDF): **~70–71% accuracy** en test  
- **RoBERTa twitter-latest** (benchmark moderno): **~40% accuracy** (predijo casi todo como negativo)  
- **Conclusión clave**: En tweets antiguos, los modelos TF-IDF + ML tradicional superan claramente a transformers actuales debido al cambio drástico en lenguaje, slang y emojis entre 2010 y 2025.

### Estructura del proyecto
.
├── sentiment_analysis_tweets.ipynb     # Notebook principal (todo el flujo)
├── training_tweets.csv                 # Dataset original (~30k tweets)
├── best_sentiment_model.pkl            # Mejor modelo clásico serializado
├── tfidf_vectorizer.pkl                # Vectorizador TF-IDF serializado
├── README.md
└── requirements.txt
text### Instalación rápida

```bash
# Entorno virtual recomendado
python -m venv venv
source venv/bin/activate          # Linux/macOS
# o en Windows: venv\Scripts\activate

pip install -r requirements.txt
Requisitos mínimos (ver requirements.txt abajo)

Python ≥ 3.9
transformers + torch (para el benchmark)
scikit-learn, pandas, nltk, matplotlib, seaborn

Uso básico (dentro del notebook)

Ejecuta las celdas en orden
Entrena los modelos clásicos → GridSearchCV con ROC-AUC
Corre el benchmark RoBERTa (puede tardar 5–15 min en CPU)
Compara métricas y matrices de confusión

### Hallazgos principales

Modelos clásicos obtienen ~0.70–0.71 accuracy (buen balance precision/recall)
Transformer moderno obtiene solo ~0.40 → colapsa prediciendo todo como negativo
Principal culpable: domain shift temporal (tweets de 2009–2011 vs entrenamiento del modelo en 2018–2022+)
Recomendación: para datos antiguos → TF-IDF + LogReg/RF. Para tweets 2023+ → transformers ganan fácilmente (~78–85%).

Próximos pasos / mejoras posibles

Probar finiteautomata/bertweet-base-sentiment-analysis (más adaptado a tweets antiguos)
Evaluar en datasets recientes (tweet_eval, SemEval 2023, etc.)
Fine-tuning ligero del transformer con subset del dataset
Interfaz simple con Gradio/Streamlit para clasificar tweets nuevos
Análisis de errores: ¿qué tipo de tweets confunde más el transformer?

Licencia
MIT License — libre para uso, modificación y distribución (atribución apreciada).
¡Pull requests y sugerencias bienvenidos!
