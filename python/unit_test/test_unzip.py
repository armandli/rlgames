import os.path
import tarfile
import glob
import gzip
import shutil
import argparse

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--file', '-f', type=str)
  args = parser.parse_args()
  return args

def unzip_data(zipfile):
  this_gz = gzip.open(zipfile)
  tar_file = zipfile[0:-3]
  this_tar = open(tar_file, 'wb')
  shutil.copyfileobj(this_gz, this_tar)
  this_tar.close()
  print('{} created'.format(tar_file))

def main():
  args = parse_args()
  unzip_data(args.file)

if __name__ == '__main__':
  main()
