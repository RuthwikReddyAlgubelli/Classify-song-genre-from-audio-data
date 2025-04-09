import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

# Suppress FutureWarning from librosa (optional)
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Function to convert audio to spectrogram
def audio_to_spectrogram(file_path, n_mels=128, hop_length=512):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
    return log_mel_spec

# 2. Load and preprocess GTZAN dataset with error handling
def load_data(data_dir, genres, target_size=(128, 128)):
    X, y = [], []
    for genre_idx, genre in enumerate(genres):
        genre_dir = os.path.join(data_dir, genre)
        for filename in os.listdir(genre_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(genre_dir, filename)
                try:
                    spectrogram = audio_to_spectrogram(file_path)
                    if spectrogram.shape[1] > target_size[1]:
                        spectrogram = spectrogram[:, :target_size[1]]
                    else:
                        spectrogram = np.pad(spectrogram, ((0, 0), (0, target_size[1] - spectrogram.shape[1])), 
                                            mode='constant')
                    X.append(spectrogram)
                    y.append(genre_idx)
                except Exception as e:
                    print(f"Skipping file {file_path} due to error: {str(e)}")
                    continue
    return np.array(X), np.array(y)

# 3. Build CNN model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 4. Main execution
def main():
    data_dir = r"C:\Projects\Classify Song Genres from Audio Data\Data\genres_original"
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    input_shape = (128, 128, 1)
    num_classes = len(genres)
    
    # Load and preprocess data
    X, y = load_data(data_dir, genres)
    if len(X) == 0:
        print("No valid audio files loaded. Check your dataset or path.")
        return
    
    X = X[..., np.newaxis]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_cnn_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_acc}")
    
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    model.save('genre_classification_model.h5')

# 5. Predict genre for new audio
def predict_genre(model, file_path, genres):
    spectrogram = audio_to_spectrogram(file_path)
    spectrogram = np.pad(spectrogram, ((0, 0), (0, 128 - spectrogram.shape[1])), 
                        mode='constant') if spectrogram.shape[1] < 128 else spectrogram[:, :128]
    spectrogram = spectrogram[np.newaxis, ..., np.newaxis]
    prediction = model.predict(spectrogram)
    predicted_genre = genres[np.argmax(prediction)]
    return predicted_genre

if __name__ == "__main__":
    main()
    
    model = tf.keras.models.load_model('genre_classification_model.h5')
    test_file = r"C:\Projects\Classify Song Genres from Audio Data\Data\genres_original\hiphop\hiphop.00004.wav"
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    predicted_genre = predict_genre(model, test_file, genres)
    print(f"Predicted genre: {predicted_genre}")
