import argparse

from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint

from rlgames.data_processor.processor import DataProcessor
from rlgames.encoders import get_encoder_by_name
from rlgames.imitation_models import small

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--size', '-s', type=int, default=19)
  parser.add_argument('--train-sample-size', '-t', type=int, default=1000)
  parser.add_argument('--test-sample-size', '-t', type=int, default=100)
  parser.add_argument('--data-directory', '-d', type=str, default='/home/armandli/rlgames/data')
  parser.add_arugment('--encoder', '-e', type=str, default='oneplane')
  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  board_sz =  args.size
  num_classes = args.size * args.size
  encoder = get_encoder_by_name(args.encoder)
  processor = DataProcessor(data_directory=args.data_directory, encoder=args.encoder)
  training_data = processor.load_data('train', num_samples=args.train_sample_size)
  test_data = processor.load_data('test', num_samples=args.test_sample_size)

if __name__ == '__main__':
  main()
