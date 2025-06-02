import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.prediction import predict_random_image


predict_random_image('models/ear_cnn_model.keras', 'data/ear_recognition_dataset/Dataset1/test')
predict_random_image('models/ear_cnn_EarVN1.0.keras', 'data/EarVN1.0/test')