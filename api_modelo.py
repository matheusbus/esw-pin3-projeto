from flask import Flask, request, jsonify, render_template, url_for
from tensorflow.keras.models import load_model
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)

# Definir o diretório de upload
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Carregar o modelo treinado
model_path = os.path.join(os.getcwd(), 'modelo_2', 'modelo', 'modelo2.keras')
print(f"Loading model from: {model_path}")
model2 = load_model(model_path)

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

@app.route('/modelo1')
def modelo1():
    return render_template('modelo1.html')

@app.route('/modelo2')
def modelo2():
    return render_template('modelo2.html')
    
@app.route('/predict2_batch', methods=['POST'])
def predict_batch():
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400
    
    images = request.files.getlist('images')
    predictions = {}

    try:
        for image in images:
            # Salvando a imagem no diretório de upload
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            
            # Pré-processamento da imagem
            processed_image = preprocess_image(image_path)
            
            # Aplicando a predição
            prediction = model2.predict(processed_image)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_label = class_labels[predicted_class_index]
            
            # Adicionando o resultado à lista de previsões
            predictions[image.filename] = predicted_label

        return jsonify(predictions)
    except Exception as e:
        print(f"Error processing the images: {e}")
        return jsonify({'error': 'Error processing the images'}), 500

if __name__ == '__main__':
    app.run(debug=True)
