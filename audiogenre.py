import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import librosa.display

def extract_mel_spectrogram(audio_file, n_mels=128):
    try:
        y, sr = librosa.load(audio_file)
        mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram_db
    except FileNotFoundError:
        print(f"File {audio_file} not found.")
        return None
    except Exception as e:
        print(f"An error occurred while processing {audio_file}: {e}")
        return None

def save_mel_spectrogram(mel_spectrogram, output_file):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# List your audio files here
audio_files = [
    "C:\Projects\Classify Song Genres from Audio Data\Data\genres_original",
]  # Replace with the actual paths to your audio files

output_dir = "C:\\Projects\\Classify Song Genres from Audio Data\\output_spectrograms"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for audio_file in audio_files:
    mel_spectrogram = extract_mel_spectrogram(audio_file)
    if mel_spectrogram is not None:
        output_file = os.path.join(output_dir, os.path.basename(audio_file) + ".png")
        save_mel_spectrogram(mel_spectrogram, output_file)



import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = 'C:\Projects\Classify Song Genres from Audio Data\Data'
batch_size = 32
img_height = 128
img_width = 128

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

from tensorflow.keras import layers, models

num_classes = len(class_names)

model = models.Sequential([
    layers.InputLayer(input_shape=(img_height, img_width, 3)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('genre_classification_model.h5')
