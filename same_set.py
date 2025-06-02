from gen_subsets import gen_subsets as data
import plot_results as p
from model import cnn_h_m_s

train_data, val_data, test_data = data('ear_recognition_dataset/Dataset1/train', 'ear_recognition_dataset/Dataset1/validation', 'ear_recognition_dataset/Dataset1/test')

model = cnn_h_m_s(13)

history = model.fit(train_data, validation_data=val_data, epochs=50)
model.save('ear_cnn_model.keras')

loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

p.plot_results(history, 'plots', 'accuracy.png', 'loss.png')