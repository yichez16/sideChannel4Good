
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.python.client import timeline

from datetime import datetime

import os
import time

# Set GPU 2 to be visible only
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# prepare datasets
t = time.time()
t_ms = int(t * 1000)
print("prepare datasets: ", t_ms)
print("-------------------------------------------------------")

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0



# CNN model with 3 cov + 2 fc
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()






# predictions = use_sess.run(use_out, {'DecodeJpeg/contents:0': image_file.file.getvalue()}, options=run_options, run_metadata=run_metadata)
# # Create the Timeline object, and write it to a json
# tl = timeline.Timeline(run_metadata.step_stats)
# ctf = tl.generate_chrome_trace_format()
# with open('timeline.json', 'w') as f:
#     f.write(ctf)



t = time.time()
t_ms = int(t * 1000)
print("start training: ", t_ms)
print("-------------------------------------------------------")

#  train model on GPU 1

try:
  with tf.device('/gpu:2'):
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
    # Get trace timeline
    # Create a TensorBoard callback

    history = model.fit(train_images, train_labels, epochs=3,
                validation_data=(test_images, test_labels),
                verbose=0

                )
except RuntimeError as e:
  print(e)

print("end training: ", t_ms)
print("-------------------------------------------------------")
