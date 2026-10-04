from flask import Flask, render_template, request, jsonify, send_file
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
import glob
import shutil
import cv2
import numpy as np
import os
import imghdr
from datetime import datetime
import threading
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')  # Default data dir

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

training_status = {
    'is_training': False,
    'progress': 0,
    'logs': [],
    'model_path': None,
    'metadata_path': None,
    'current_epoch': 0,
    'total_epochs': 0,
    'accuracy': 0,
    'loss': 0,
    'class_names': [],
    'image_size': 256
}

loaded_model = None
loaded_metadata = None

def auto_load_latest_model():
    global loaded_model, loaded_metadata
    
    model_files = glob.glob(os.path.join(MODELS_DIR, 'trained_model_*.h5'))
    if not model_files:
        print("No existing models found.")
        return
    
    # Get latest model
    latest_model = max(model_files, key=os.path.getctime)
    timestamp = os.path.basename(latest_model).replace('trained_model_', '').replace('.h5', '')
    metadata_file = os.path.join(MODELS_DIR, f'metadata_{timestamp}.json')
    
    if os.path.exists(metadata_file):
        try:
            loaded_model = keras.models.load_model(latest_model)
            with open(metadata_file, 'r') as f:
                loaded_metadata = json.load(f)
            print(f"Loaded latest model: {latest_model}")
            training_status['model_path'] = latest_model
            training_status['metadata_path'] = metadata_file
            training_status['class_names'] = loaded_metadata['class_names']
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print("No matching metadata found for latest model.")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}


def add_log(message, log_type='info'):
    training_status['logs'].append({
        'message': message,
        'type': log_type,
        'time': datetime.now().strftime('%H:%M:%S')
    })


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class TrainingCallback(Callback):

    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        training_status['current_epoch'] = epoch + 1
        training_status['progress'] = ((epoch + 1) / self.total_epochs) * 80 + 20
        training_status['accuracy'] = float(logs.get('accuracy', 0))
        training_status['loss'] = float(logs.get('loss', 0))

        add_log(
            f"Epoch {epoch+1}/{self.total_epochs} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f}"
        )


def clean_dataset(data_dir):
    removed_count = 0

    for image_class in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, image_class)

        if not os.path.isdir(class_dir):
            continue

        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)

            try:
                tip = imghdr.what(image_path)

                if tip is None or tip not in ['jpg', 'jpeg', 'png', 'bmp']:
                    print(f"Removing invalid image: {image_path}")
                    os.remove(image_path)
                    removed_count += 1

            except Exception as e:
                print(f"Error checking {image_path}: {e}")
                try:
                    os.remove(image_path)
                    removed_count += 1
                except:
                    pass

    print(f"Cleaned {removed_count} invalid images")
    return removed_count


def count_images_and_classes(data_dir):

    total_images = 0
    classes = []

    for image_class in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, image_class)

        if not os.path.isdir(class_dir):
            continue

        classes.append(image_class)

        image_files = [
            f for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f))
        ]

        total_images += len(image_files)

    return total_images, len(classes), classes


def build_model(num_classes, image_size):

    model = Sequential([
        Conv2D(16, (3, 3), activation='relu',
               input_shape=(image_size, image_size, 3)),
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    return model


def train_model_async(config):

    global loaded_model, loaded_metadata

    try:

        training_status['is_training'] = True

        data_dir = config['data_dir']
        epochs = config['epochs']
        batch_size = config['batch_size']
        image_size = config['image_size']

        add_log("Training started")

        total_images, num_classes, class_names = count_images_and_classes(data_dir)

        training_status['class_names'] = class_names

        data = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            validation_split=0.2,
            subset='training',
            seed=123
        )

        val_data = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            validation_split=0.2,
            subset='validation',
            seed=123
        )

        data = data.cache().shuffle(1000).repeat().prefetch(tf.data.AUTOTUNE)
        val_data = val_data.cache().prefetch(tf.data.AUTOTUNE)

        model = build_model(num_classes, image_size)

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        callback = TrainingCallback(epochs)

        model.fit(data, epochs=epochs, validation_data=val_data, callbacks=[callback])

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        saved_model_dir = os.path.join(MODELS_DIR, f"trained_model_{timestamp}")
        model.save(saved_model_dir)

        metadata = {
            "class_names": class_names,
            "image_size": image_size,
            "num_classes": num_classes
        }

        metadata_path = os.path.join(MODELS_DIR, f"metadata_{timestamp}.json")

        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        training_status['model_path'] = saved_model_dir

        loaded_model = model
        loaded_metadata = metadata

        add_log("Training completed")

    except Exception as e:

        add_log(str(e), "error")

    finally:

        training_status['is_training'] = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/train', methods=['POST'])
def train():

    if training_status['is_training']:
        return jsonify({'error': 'Training already running'}), 400

    if not request.is_json:
        return jsonify({'error': 'Invalid JSON'}), 400

    config = request.json
    required_keys = ['data_dir', 'epochs', 'batch_size', 'image_size']
    if not all(key in config for key in required_keys):
        return jsonify({'error': 'Missing required config fields'}), 400

    if not os.path.exists(config['data_dir']):
        return jsonify({'error': f'Data directory not found: {config["data_dir"]}'}), 400

    if not (1 <= config['epochs'] <= 100):
        return jsonify({'error': 'Epochs must be between 1 and 100'}), 400

    if not (1 <= config['batch_size'] <= 128):
        return jsonify({'error': 'Batch size must be between 1 and 128'}), 400

    if not (32 <= config['image_size'] <= 512):
        return jsonify({'error': 'Image size must be between 32 and 512'}), 400

    thread = threading.Thread(target=train_model_async, args=(config,))
    thread.daemon = True
    thread.start()

    return jsonify({'message': 'Training started'})


@app.route('/api/dataset/info', methods=['POST'])
def dataset_info():
    data_dir = request.json.get('data_dir', DATA_DIR)
    try:
        total_images, num_classes, class_names = count_images_and_classes(data_dir)
        return jsonify({
            'total_images': total_images,
            'num_classes': num_classes,
            'class_names': class_names
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/list')
def models_list():
    try:
        model_files = glob.glob(os.path.join(MODELS_DIR, 'trained_model_*.h5'))
        models = []
        for model_file in model_files:
            timestamp = os.path.basename(model_file).replace('trained_model_', '').replace('.h5', '')
            metadata_file = os.path.join(MODELS_DIR, f'metadata_{timestamp}.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                models.append({
                    'path': model_file,
                    'timestamp': timestamp,
                    'num_classes': metadata['num_classes'],
                    'image_size': metadata['image_size'],
                    'class_names': metadata['class_names'][:5] + ['...'] if len(metadata['class_names']) > 5 else metadata['class_names']
                })
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/load', methods=['POST'])
def load_model():
    try:
        data = request.json
        model_path = data['model_path']
        timestamp = os.path.basename(model_path).replace('trained_model_', '').replace('.h5', '')
        metadata_file = os.path.join(MODELS_DIR, f'metadata_{timestamp}.json')
        
        if not os.path.exists(metadata_file):
            return jsonify({'error': 'Metadata not found'}), 404
        
        global loaded_model, loaded_metadata
        loaded_model = keras.models.load_model(model_path)
        with open(metadata_file, 'r') as f:
            loaded_metadata = json.load(f)
        
        training_status['model_path'] = model_path
        training_status['metadata_path'] = metadata_file
        training_status['class_names'] = loaded_metadata['class_names']
        training_status['image_size'] = loaded_metadata['image_size']
        
        return jsonify({
            'success': True,
            'class_names': loaded_metadata['class_names'],
            'image_size': loaded_metadata['image_size']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def status():
    return jsonify(training_status)


@app.route('/api/predict', methods=['POST'])
def predict():
    global loaded_model, loaded_metadata

    if loaded_model is None:
        return jsonify({'error': 'Model not loaded'})

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOADS_DIR, filename)

    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

    image = cv2.imread(filepath)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    try:
        image = cv2.resize(image, (loaded_metadata['image_size'], loaded_metadata['image_size']))
        image = np.expand_dims(image, axis=0) / 255.0
        predictions = loaded_model.predict(image)[0]
        
        confidence = float(np.max(predictions))
        idx = np.argmax(predictions)
        predicted_class = loaded_metadata['class_names'][idx]
        
        all_predictions = {cls: float(pred) for cls, pred in zip(loaded_metadata['class_names'], predictions)}

        # Cleanup uploaded file
        os.remove(filepath)

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.2%}",
            "all_predictions": all_predictions
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/download')
def download():

    if training_status['model_path']:
        return send_file(training_status['model_path'], as_attachment=True, download_name=os.path.basename(training_status['model_path']))

    return jsonify({'error': 'No model available'})


if __name__ == "__main__":
    print("ML Training Platform Running")
    auto_load_latest_model()
    app.run(debug=True, port=5000)
