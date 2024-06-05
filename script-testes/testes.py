import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def listar_arquivos_no_diretorio():
    cwd = os.getcwd()
    print(f"Diretório atual: {cwd}")
    arquivos = os.listdir(cwd)
    print("Arquivos no diretório atual:")
    for arquivo in arquivos:
        print(arquivo)

listar_arquivos_no_diretorio()

modelo1_path = os.path.join(os.getcwd(), 'modelo_2', 'modelo', 'modelo2.keras')
print(f"Loading model from: {modelo1_path}")
if not os.path.exists(modelo1_path):
    raise FileNotFoundError(f"O arquivo {modelo1_path} não foi encontrado.")

modelo1 = load_model(modelo1_path)

modelo2_path = os.path.join(os.getcwd(), 'modelo_2', 'modelo', 'modelo2.keras')
print(f"Loading model from: {modelo2_path}")
if not os.path.exists(modelo2_path):
    raise FileNotFoundError(f"O arquivo {modelo2_path} não foi encontrado.")
modelo2 = load_model(modelo2_path)

dataset_teste_dir = os.path.join(os.getcwd(), 'script-testes', 'imagens_teste')

datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = datagen.flow_from_directory(
    dataset_teste_dir,
    target_size=(224, 224),  
    batch_size=32,
    class_mode='binary',     
    shuffle=False           
)

y_test = test_generator.classes

y_pred_modelo1_prob = modelo1.predict(test_generator)
y_pred_modelo1 = np.argmax(y_pred_modelo1_prob, axis=-1)  

y_pred_modelo2_prob = modelo2.predict(test_generator)
y_pred_modelo2 = np.argmax(y_pred_modelo2_prob, axis=-1) 

def calcular_metricas(y_true, y_pred):
    return {
        'Acurácia': accuracy_score(y_true, y_pred),
        'Precisão': precision_score(y_true, y_pred, average='macro'),
        'Recall': recall_score(y_true, y_pred, average='macro'),
        'F1 Score': f1_score(y_true, y_pred, average='macro')
    }

metricas_modelo1 = calcular_metricas(y_test, y_pred_modelo1)
metricas_modelo2 = calcular_metricas(y_test, y_pred_modelo2)

print("Resultados do Modelo 1:")
for metrica, valor in metricas_modelo1.items():
    print(f"{metrica}: {valor:.4f}")

print("\nResultados do Modelo 2:")
for metrica, valor in metricas_modelo2.items():
    print(f"{metrica}: {valor:.4f}")
