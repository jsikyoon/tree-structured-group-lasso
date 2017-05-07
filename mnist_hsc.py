from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import input_data
import hsc
import numpy as np
import time

FLAGS = None


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)
  tr_data, tr_label = mnist.train.next_batch(mnist.train._num_examples);
  # Dictionary Initialization
  M=len(tr_data[0]);
  D=hsc.dict_initializer(M,FLAGS.P);
  # Learning
  lr=FLAGS.learning_rate; pre_mse=10;
  for i in range(1,FLAGS.max_steps+1):
    # Data Shuffle
    idx=range(len(tr_data));np.random.shuffle(idx);
    batch=tr_data[idx[:FLAGS.batch_num]].transpose();
    # Learning Rate Decay
    if(i%FLAGS.decay_num==0):
      lr=lr/float(FLAGS.decay_rate);
    # Sparse Coding
    A=hsc.sparse_coding(D,batch,FLAGS);
    # Dictionary Learning
    D=hsc.dictionary_learning(D,batch,A,lr,FLAGS);
    loss=np.linalg.norm(np.matmul(D,A)-batch,axis=0);mse=np.mean(loss);
    print(str(i)+"th MSE: "+str(mse));
    mse_diff=abs(mse-pre_mse);
    if(mse_diff<FLAGS.mse_diff_threshold):
      print("Learning Done");
      exit(1);
    pre_mse=mse;
  print("Max Iterations Done");

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--learning_rate', type=float, default=1,
                      help='Initial learning rate')
  parser.add_argument('--decay_num', type=int, default=10,
                      help='Every decay_num, learning rate is decayed')
  parser.add_argument('--decay_rate', type=int, default=2,
                      help='Decay Late')
  parser.add_argument('--max_steps', type=float, default=100,
                      help='Max Iterations')
  parser.add_argument('--mse_diff_threshold', type=int, default=0.001,
                      help='The threshold of Loss Diff.')
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--P', type=int, default=5000,
                      help='The Number of dictionary atoms')
  parser.add_argument('--batch_num', type=int, default=20,
                      help='The Number of data per each batch')
  parser.add_argument('--sc_max_steps', type=int, default=100,
                      help='Sparse Coding Max Iterations')
  parser.add_argument('--sc_mse_diff_threshold', type=int, default=0.001,
                      help='Sparse Coding threshold of Loss Diff');
  parser.add_argument('--sc_lambda', type=int, default=0.1,
                      help='The ratio between loss and regularization terms');
  FLAGS, unparsed = parser.parse_known_args()
  train();
