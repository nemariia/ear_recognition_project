import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.gen_subsets import gen_subsets as data
import plots.plot_results as p
from models.model import cnn_h_m_s


train_data, val_data, test_data = data('data/EarVN1.0/train', 'data/EarVN1.0/val', 'data/EarVN1.0/test')

model = cnn_h_m_s(164)

history = model.fit(train_data, validation_data=val_data, epochs=100)
model.save('models/ear_cnn_EarVN1.0.keras')

loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

p.plot_results(history, 'plots', 'accuracy_EarVN1.0.png', 'loss_EarVN1.0.png')