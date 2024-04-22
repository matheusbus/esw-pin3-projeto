from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Carregar o modelo treinado
model_path = os.getcwd() + '/modelo_1/modelo1.keras'
print(model_path)
model = load_model(model_path)

# Mapear as classes para rótulos legíveis pelo humano
class_labels = {0: 'Morango', 1: 'Pêssego', 2: 'Romã'}

# Rota para receber a imagem e fazer a previsão
@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    
    # Pré-processamento da imagem
    image = Image.open(image_file)
    image = image.resize((300, 300))  
    image = np.array(image) / 255.0    # Normalizar os pixels
    
    # Aplicando a predição
    predictions = model.predict(np.expand_dims(image, axis=0))
    print(predictions)
    
    predicted_class = np.argmax(predictions)
    print(predicted_class)
    predicted_label = class_labels[predicted_class]

    print('Classificado como: ', predicted_label)
    
    return jsonify({'class': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
