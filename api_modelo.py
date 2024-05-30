from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)

# Carregar o modelo treinado
model_path = os.path.join(os.getcwd(), 'modelo_2', 'modelo', 'modelo2.keras')
print(f"Loading model from: {model_path}")
model = load_model(model_path)

# Mapear as classes para rótulos legíveis pelo humano
class_labels = {0: 'Morango', 1: 'Pêssego', 2: 'Romã'}

def preprocess_image(image_path, target_size=(300, 300)):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/modelo2')
def modelo2():
    return render_template('modelo2.html')

@app.route('/predict2', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    
    try:
        # Salvando a imagem temporariamente
        image_path = os.path.join(os.getcwd(), 'temp_image.jpg')
        image_file.save(image_path)
        
        # Pré-processamento da imagem
        processed_image = preprocess_image(image_path)
        
        # Aplicando a predição
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels[predicted_class_index]
        
        print(f'Classificado como: {predicted_label}')
        
        # Removendo a imagem temporária
        os.remove(image_path)
        
        return jsonify({'class': predicted_label})
    except Exception as e:
        print(f"Error processing the image: {e}")
        return jsonify({'error': 'Error processing the image'}), 500

if __name__ == '__main__':
    app.run(debug=True)
