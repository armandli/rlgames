import argparse
import h5py

from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

from rlgames.data_processor.parallel_processor import DataProcessor
from rlgames.encoders import get_encoder_by_name
from rlgames.agents.predict import DeepLearningAgent

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--board-size', '-s', type=int, default=19)
  parser.add_argument('--train-size', '-t', type=int, default=100)
  parser.add_argument('--test-size', '-z', type=int, default=10)
  parser.add_argument('--data-dir', '-d', type=str,
          default='/home/armandli/rlgames/data')
  parser.add_argument('--encoder', '-e', type=str, default='oneplane')
  parser.add_argument('--model-size', '-m', type=str, default='small')
  parser.add_argument('--epoch', '-p', type=int, default=100)
  parser.add_argument('--batchsize', '-b', type=int, default=64)
  parser.add_argument('--optimizer', '-r', type=str, default='sgd')
  parser.add_argument('--output', '-o', type=str,
          default='sl')
  args = parser.parse_args()
  return args

def load_model(args, encoder):
  if args.model_size == 'small':
    from rlgames.imitation_models.small import layers
  elif args.model_size == 'medium':
    from rlgames.imitation_models.medium import layers
  elif args.model_size == 'large':
    from rlgames.imitation_models.large import layers
  else:
    print('Unknown model type: {}. Use small.'.format(args.model_size))
    from rlgames.imitation_models.small import layers

  model = Sequential()
  input_shape =  (encoder.num_planes, args.board_size, args.board_size)
  network_layers = layers(input_shape)
  for layer in network_layers:
    model.add(layer)
  model.add(Dense(args.board_size * args.board_size, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer=args.optimizer, metrics=['accuracy'])
  return model

def main():
  args = parse_args()
  num_classes = args.board_size *args.board_size
  encoder = get_encoder_by_name(args.encoder,args.board_size)
  processor = DataProcessor(data_directory=args.data_dir,
          encoder=args.encoder)
  print('Loading data')
  TrainX, TrainY = processor.load_data('train', num_samples=args.train_size)
  TestX, TestY = processor.load_data('test', num_samples=args.test_size)
  model = load_model(args, encoder)
  checkpoint_dir = args.data_dir + '/checkpoints/'
  print('Begin training')
  model.fit(TrainX, TrainY, args.batchsize, args.epoch, callbacks =
          [ModelCheckpoint(checkpoint_dir + 'sl_{}_{}_{}_'.format(args.encoder, args.model_size, args.board_size) + 'epoch_{epoch}.h5')])
  print('Training complete')
  validation = model.evaluate(TestX, TestY, args.batchsize)
  print('Validation Loss: {}'.format(validation[0]))
  print('Vlidationn Accuracy: {}'.format(validation[1]))
  print('Saving model')
  h5file = h5py.File(args.data_dir + '/agents/' + args.output + '_' + args.model_size + '_' + args.encoder + '_' + str(args.board_size) + '.h5', 'w')
  agent = DeepLearningAgent(model, encoder)
  agent.serialize(h5file)

if __name__ == '__main__':
  main()
