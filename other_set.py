import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# Data generators
train_gen = ImageDataGenerator(rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

val_test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    'EarVN1.0/train',
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_data = val_test_gen.flow_from_directory(
    'EarVN1.0/val',
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_data = val_test_gen.flow_from_directory(
    'EarVN1.0/test',
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)


model = Sequential([
    Input(shape=(128, 128, 1)),
    Conv2D(16, kernel_size=(2, 2), strides=1, activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, kernel_size=(2, 2), strides=1, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(2, 2), strides=1, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(500, activation='relu'),
    # Dropout(0.4),
    Dense(164, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Use sparse_categorical_crossentropy if labels are integers
    metrics=['accuracy']
)

history = model.fit(train_data, validation_data=val_data, epochs=50)
model.save('ear_cnn_model_updated.keras')

loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy.png')  # Save to file

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss.png')
