# =========================
# 1 Import des bibliothèques
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# =========================
# 2 Chargement du dataset
# =========================
data = pd.read_csv("hdfs.log_templates.csv")
print("Aperçu des données :")
print(data.head())

# Vérifier les colonnes
print("\nColonnes du dataset :", data.columns)

# =========================
# 3 Exploration des templates
# =========================
# Compter les occurrences de chaque EventTemplate
template_counts = data['EventTemplate'].value_counts()
print("\nTop 10 des templates les plus fréquents :")
print(template_counts.head(10))

# Visualisation
plt.figure(figsize=(10,6))
sns.barplot(x=template_counts.head(10).values, 
            y=template_counts.head(10).index, 
            palette="viridis")
plt.title("Top 10 des EventTemplates les plus fréquents")
plt.xlabel("Nombre d'occurrences")
plt.ylabel("EventTemplate")
plt.show()

# =========================
# 4 Feature Engineering
# =========================
# Convertir EventTemplate en features numériques (One-Hot Encoding)
features = pd.get_dummies(data['EventTemplate'])

print("\nExemple de features :")
print(features.head())

# 5 Séparation des données train/test
# =========================
X_train, X_test = train_test_split(features, test_size=0.3, random_state=42)
print(f"\nTaille du train set : {X_train.shape}")
print(f"Taille du test set  : {X_test.shape}")

# =========================
# 6 Détection d'anomalies avec Isolation Forest
# =========================
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(features)

# Prédiction : -1 = anomalie, 1 = normal
data['Anomaly'] = model.predict(features)
data['Anomaly'] = data['Anomaly'].map({1:'Normal', -1:'Anomaly'})

print("\nExemple de détection d'anomalies :")
print(data[['EventTemplate','Anomaly']].head(20))

# =========================
# 7 Analyse des anomalies
# =========================
anomaly_counts = data['Anomaly'].value_counts()
print("\nNombre de Normal vs Anomalie :")
print(anomaly_counts)

# Visualisation
plt.figure(figsize=(6,4))
sns.countplot(x='Anomaly', data=data, palette="Set2")
plt.title("Nombre d'événements normaux vs anomalies")