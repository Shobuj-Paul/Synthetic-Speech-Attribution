import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class MFCCFeatureExtraction():
  def __init__(self, tracks, labels):
    self.tracks = tracks
    self.labels = labels

  def getfeatures(self, filter = 20):
    return [librosa.feature.mfcc(audio, sr=16000, n_mfcc=filter) for audio in self.tracks]

  def visualize(self, index):
    fig, ax = plt.subplots(figsize = (8, 5))
    data= self.getfeatures()
    img = librosa.display.specshow(data[index], x_axis='time', ax = ax)
    fig.colorbar(img)
    ax.set(title = "MFCC")

  def featuredata(self):
      scaled_features = []
      features = self.getfeatures()

      for data in features:
        mfcc = np.mean(data.T, axis = 0)
        scaled_features.append(mfcc)
      
      one_hot_labels = tf.keras.utils.to_categorical(self.labels, num_classes = 5) 
      return (np.array(scaled_features), one_hot_labels)
