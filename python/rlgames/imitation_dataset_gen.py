from rlgames.data_processor.processor import DataProcessor
from rlgames.encoders import get_encoder_by_name
from rlgames.imitation_models import small

from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint

board_sz = 19
num_classes = board_sz * board_sz
num_games = 100
data_dir='/home/armandli/rlgames/data'

encoder = get_encoder_by_name('elevenplane', board_sz)
processor = DataProcessor(data_directory=data_dir, encoder = 'elevenplane')

training_data = processor.load_data('train', num_samples=1024)
test_data = processor.load_data('test', num_samples=100)
