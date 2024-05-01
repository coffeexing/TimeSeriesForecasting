import tensorflow as tf
import keras
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import tensorflow as tf

tensorflow_version=tf.__version__
gpu_avaliable=tf.test.is_gpu_available()

print(tensorflow_version)

a=tf.constant([1.0,2.0],name='a')
b=tf.constant([1.0,2.0],name='b')
c=tf.add(a,b,name='add')
print(c)
