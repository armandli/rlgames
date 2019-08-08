import os
import argparse
import h5py

from rlgames.kerasutil import load_model_from_hdf5_group, save_model_to_hdf5_group

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', '-p', type=str, required=True)
  parser.add_argument('--out', '-o', type=str, required=True)
  return parser.parse_args()

def main():
  args = parse_args()
  if not os.path.isfile(args.path):
    raise ValueError('File {} does not exist!'.format(args.path))
  with h5py.File(args.path, 'r') as h5file:
    model = load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    print(encoder_name)
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    print('board width {} and height {}'.format(board_width, board_height))
    with h5py.File(args.out, 'w') as outfile:
      outfile.create_group('encoder')
      outfile['encoder'].attrs['name'] = encoder_name
      outfile['encoder'].attrs['board_sz'] = board_width
      outfile.create_group('model')
      save_model_to_hdf5_group(model, outfile['model'])
  print('Model saved to {}'.format(args.out))

if __name__ == '__main__':
  main()
