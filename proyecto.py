import pandas as pd 
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import gdown

# Descargar datos desde Google Drive
file_id = "1oFS83T-O4ZKg5c5u4ytxSynHxBJgSfJx"  # Reemplaza con el ID real de tu archivo
output = "creditcard.csv"
url = f"https://drive.google.com/uc?id={file_id}"
print("Descargando datos desde Google Drive...")
gdown.download(url, output, quiet=False)
print("Datos descargados correctamente.")

# Cargar datos
df = pd.read_csv("creditcard.csv")
print("Datos cargados correctamente.")

# Preprocesamiento
print("Normalizando la columna 'Amount'...")
df['Amount'] = StandardScaler().fit_transform(df[['Amount']])

print("Preparando variables de entrada y salida...")
X = df.drop(columns=['Class', 'Time'])
y = df['Class']

# Dividir en entrenamiento y prueba
print("Dividiendo los datos en entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Usar solo el 10% de los datos de entrenamiento para agilizar
X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=42)
print(f"Entrenando con {len(X_train_sample)} muestras...")

# Aplicar SMOTE para balancear las clases
print(f"Antes de SMOTE: {dict(pd.Series(y_train_sample).value_counts())}")
smote = SMOTE(random_state=42)
X_train_sample, y_train_sample = smote.fit_resample(X_train_sample, y_train_sample)
print(f"Después de SMOTE: {dict(pd.Series(y_train_sample).value_counts())}")

# Entrenar modelo Random Forest con menos árboles para agilizar
print("Entrenando modelo Random Forest...")
start_time = time.time()

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train_sample, y_train_sample)

end_time = time.time()
print(f"Modelo entrenado en {end_time - start_time:.2f} segundos.")

# Hacer predicciones
print("Realizando predicciones...")
y_pred = model.predict(X_test)

# Evaluar modelo
print("Evaluando modelo...")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

# Graficar la matriz de confusión
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fraude', 'Fraude'], yticklabels=['No Fraude', 'Fraude'])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

# Graficar la distribución de predicciones
plt.figure(figsize=(6,4))
sns.countplot(x=y_pred, hue=y_pred, palette='pastel', legend=False)
plt.xticks([0,1], ['No Fraude', 'Fraude'])
plt.xlabel("Clase Predicha")
plt.ylabel("Cantidad de Predicciones")
plt.title("Distribución de Predicciones")
plt.show()
