import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.gen_subsets import gen_subsets as data
import plots.plot_results as p
from models.model import cnn_h_m_s


train_data, val_data, test_data = data('data/ear_recognition_dataset/Dataset1/train', 'data/ear_recognition_dataset/Dataset1/validation', 'data/ear_recognition_dataset/Dataset1/test')

model = cnn_h_m_s(13)

history = model.fit(train_data, validation_data=val_data, epochs=10)
#model.save('models/ear_cnn_model.keras')

loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

#p.plot_results(history, 'plots', 'accuracy.png', 'loss.png')