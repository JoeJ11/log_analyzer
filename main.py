import os

import lstm_analyzer
import config
from generator import Generator

tmp_path = os.path.abspath(os.getcwd())
generator = Generator()
os.chdir(tmp_path)

analyzer = lstm_analyzer.LSTM_Analyzer(generator)
analyzer._generate_feature()
analyzer._train_model()