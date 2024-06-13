import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D


def load_and_preprocess_image(image_path, target_size=(128, 128)):
    try:
        image = Image.open(image_path)
        image = image.resize(target_size)
        image = image.convert('RGB')  # Преобразование изображения в RGB
        image = np.array(image) / 255.0
        return image
    except FileNotFoundError:
        print(f"Файл не найден: {image_path}")
        return None


def load_images_from_directory(directory, target_size=(128, 128)):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = load_and_preprocess_image(image_path, target_size)
            if image is not None:
                images.append(image)
    return np.array(images)


def visualize_results(original, encoded, decoded):
    plt.figure(figsize=(15, 5))
    # Original image
    ax = plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    # Encoded image (average over channels)
    ax = plt.subplot(1, 3, 2)
    encoded_avg = np.mean(encoded, axis=-1)  # Среднее значение по всем каналам
    plt.imshow(encoded_avg, cmap='viridis')
    plt.title("Encoded")
    plt.axis("off")

    # Decoded image
    ax = plt.subplot(1, 3, 3)
    plt.imshow(decoded)
    plt.title("Decoded")
    plt.axis("off")

    plt.show()


def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    # Autoencoder model
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder


# Укажите путь к директории с изображениями для обучения
image_directory = 'C:\\Users\\Андрей\\Desktop\\Koder_incoder\\training_image'

# Загрузите и предобработайте изображения
images = load_images_from_directory(image_directory)
if len(images) == 0:
    raise ValueError("Не удалось загрузить изображения. Проверьте путь к директории и наличие изображений.")

# Определите архитектуру автоэнкодера
input_shape = (128, 128, 3)
autoencoder = build_autoencoder(input_shape)

# Обучите модель
autoencoder.fit(images, images, epochs=3500, batch_size=16, shuffle=True)

# Сохраните модель
autoencoder.save('autoencoder_model.h5')

# Укажите путь к оригинальному изображению
image_path = 'C:\\Users\\Андрей\\Desktop\\Koder_incoder\\original_images\\image1.jpg'

# Загрузите и предобработайте изображение
image = load_and_preprocess_image(image_path)
if image is None:
    raise ValueError("Не удалось загрузить изображение. Проверьте путь к файлу.")

# Прогоните изображение через кодер
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=3).output)
encoded_image = encoder.predict(np.expand_dims(image, axis=0))[0]

# Создайте директорию для сжатых изображений, если она не существует
os.makedirs('compressed_images', exist_ok=True)

# Сохраните сжатое изображение
encoded_image_path = os.path.join('compressed_images', 'encoded_image.npy')
np.save(encoded_image_path, encoded_image)

# Загрузите сжатое изображение
encoded_image = np.load(encoded_image_path)

# Восстановите изображение с использованием автоэнкодера
decoded_image = autoencoder.predict(np.expand_dims(image, axis=0))[0]

# Визуализация результатов
visualize_results(image, encoded_image, decoded_image)