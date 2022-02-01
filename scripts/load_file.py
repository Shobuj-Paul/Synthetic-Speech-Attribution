import os
import numpy as np
import librosa
import pandas as pd
import tensorflow as tf

class LoadFile(tf.keras.utils.Sequence):
  def __init__(self, data_set_path):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_set = pd.read_csv(os.path.join(dir_path, data_set_path))
        self.track, self.label = data_set['track'], data_set['algorithm']

  def __len__(self):
        return len(self.track)

  def __getitem__(self, count = None):
      batch_x = self.track
      batch_y = self.label
      if count:
          batch_x = self.track[:count]
          batch_y = self.label[:count]

      x = np.array([librosa.load(file_name, sr =16000) for file_name in batch_x], dtype=object)
      y = np.array([label for label in batch_y])
      x = x[:,0]

      return  x, y