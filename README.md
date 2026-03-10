Descripción
Proyecto de clasificación binaria de sentimiento (positivo / negativo) sobre un dataset clásico de ~30.000 tweets en inglés (Sentiment140, 2009–2011).
El objetivo principal es determinar si los modelos clásicos bien calibrados siguen siendo competitivos — o superiores — frente a transformers modernos cuando existe domain shift temporal, es decir, cuando el lenguaje del dataset difiere significativamente del usado para entrenar el modelo preentrenado.

Hallazgos Principales
ModeloAccuracyObservaciónTF-IDF + Logistic Regression~71%Mejor modelo — balance sólido precision/recallTF-IDF + Random Forest~70%Resultado competitivoTF-IDF + BernoulliNB~68%Baseline clásicoRoBERTa (twitter-latest)~40%Colapsa prediciendo casi todo como negativo
Conclusión clave: En tweets de 2009–2011, los modelos TF-IDF + ML tradicional superan en ~30 puntos porcentuales a RoBERTa out-of-the-box, debido al fuerte domain shift temporal en lenguaje, slang y contexto emocional.

Metodología
Preprocesamiento

Limpieza de texto (URLs, menciones, caracteres especiales)
Remoción de stopwords y lematización con NLTK
Vectorización TF-IDF con n-gramas (1–2)

Modelos clásicos

Logistic Regression, Random Forest, BernoulliNB
Optimización con GridSearchCV por ROC-AUC

Benchmark transformer

cardiffnlp/twitter-roberta-base-sentiment-latest via Hugging Face
Mapeo de clases: neutral/positive → positive

Métricas evaluadas

Accuracy, Precision, Recall, F1-score
ROC-AUC, Log Loss, Matriz de confusión


Estructura del Proyecto
├── analisis_sentimientos.ipynb   # Notebook principal — flujo completo
├── training_tweets.csv           # Dataset original (~30k tweets)
├── best_sentiment_model.pkl      # Mejor modelo clásico serializado
├── tfidf_vectorizer.pkl          # Vectorizador TF-IDF serializado
├── Resumen_ejecutivo.txt         # Resumen técnico y conclusiones
├── requirements.txt              # Dependencias del proyecto
└── README.md

Instalación
bash# Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Instalar dependencias
pip install -r requirements.txt
Requisitos mínimos:

Python ≥ 3.9
scikit-learn, pandas, nltk, matplotlib, seaborn
transformers + torch (para el benchmark RoBERTa)


Uso

Ejecutar las celdas del notebook analisis_sentimientos.ipynb en orden
El pipeline entrena los modelos clásicos con GridSearchCV
El benchmark RoBERTa se ejecuta al final (puede tardar 5–15 min en CPU)
Las métricas y matrices de confusión se generan automáticamente al final


Conclusiones y Recomendaciones

Para datos históricos (pre-2015): TF-IDF + LogReg/RF son superiores y mucho más eficientes
Para tweets recientes (2023+): transformers superan fácilmente el 80% de accuracy
El domain shift temporal es el principal factor que explica el colapso de RoBERTa en este dataset

Próximos pasos:

Evaluar finiteautomata/bertweet-base-sentiment-analysis (más adaptado a tweets antiguos)
Fine-tuning ligero del transformer con un subset del dataset
Interfaz simple con Gradio o Streamlit para clasificar tweets en tiempo real
Análisis de errores: identificar qué tipo de tweets confunde más a cada modelo


Tecnologías
Python · scikit-learn · NLTK · Hugging Face Transformers · Pandas · Matplotlib · Seaborn

Licencia
MIT License — libre para uso, modificación y distribución (atribución apreciada).
Pull requests y sugerencias son bienvenidos.
