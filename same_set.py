import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from gen_subsets import gen_subsets as data
import plot_results as p


train_data, val_data, test_data = data('ear_recognition_dataset/Dataset1/train', 'ear_recognition_dataset/Dataset1/validation', 'ear_recognition_dataset/Dataset1/test')

model = Sequential([
    Input(shape=(128, 128, 1)),
    Conv2D(16, kernel_size=(2, 2), strides=1, activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),

    Conv2D(32, kernel_size=(2, 2), strides=1, activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),

    Conv2D(64, kernel_size=(2, 2), strides=1, activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),

    Flatten(),

    Dense(500, activation='relu'),
    Dense(13, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train_data, validation_data=val_data, epochs=50)
model.save('ear_cnn_model.keras')

loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

p.plot_results(history, 'plots', 'accuracy.png', 'loss.png')