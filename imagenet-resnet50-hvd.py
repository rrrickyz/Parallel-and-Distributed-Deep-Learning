# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

# Helper libraries
import time
import os
import numpy as np
import matplotlib.pyplot as plt
# import argparse

# Horovod: import and initialize
import horovod.tensorflow.keras as hvd
hvd.init()
# parser = argparse.ArgumentParser()
# parser.add_argument(‘ -- epochs’, type=int, default=50)
# parser.add_argument(‘ -- batch_size’, type=int, default=32)

# args = parser.parse_args()
# batch_size = args.batch_size
# epochs = args.epochs

batch_size = 32
epochs = 50

if hvd.rank() == 0:
    print('Using Tensorflow version:', tf.__version__,
          'Keras version:', tf.keras.__version__,
          'backend:', tf.keras.backend.backend())
    print('Using Horovod with', hvd.size(), 'workers')

# Horovod: pin GPU to be used to process local rank (one GPU per process)
# return a list of physical GPUs visible to the host runtime
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(
                 gpus[hvd.local_rank()], 'GPU')

# Imagenet dataset
# get imagenet labels
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
dataset_dir = '/scratch/project_2006142/'  # directory where you downloaded the tar files to
temp_dir = '/scratch/project_2006142/temp/'  # a temporary directory where the data will be stored intermediately
# ILSVRC2012_img_train.tar
# Construct a tf.data.Dataset
download_config = tfds.download.DownloadConfig(
                      extract_dir=os.path.join(temp_dir, 'extracted'),
                      manual_dir=dataset_dir
                  )
download_and_prepare_kwargs = {
    'download_dir': os.path.join(temp_dir, 'downloaded'),
    'download_config': download_config,
}
(train_dataset, validation_dataset) = tfds.load('imagenet2012',
               data_dir=os.path.join(temp_dir, 'data'),
               split=['train', 'validation'],
               shuffle_files=True,
               download=True,
               as_supervised=True,
               download_and_prepare_kwargs=download_and_prepare_kwargs)

def resize_with_crop(image, label):
    i = image
    i = tf.cast(i, tf.float32)
    i = tf.image.resize_with_crop_or_pad(i, 224, 224)
    #i = tf.keras.applications.mobilenet_v2.preprocess_input(i)
    return (i, label)

# Preprocess the images
train_dataset = train_dataset.map(resize_with_crop, num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.map(resize_with_crop, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
train_dataset = train_dataset.shard(hvd.size(), hvd.rank())
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(batch_size, drop_remainder=True)
validation_dataset = validation_dataset.shard(hvd.size(), hvd.rank())
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

epoch = 50//hvd.size()

# the model
inputs = keras.Input(shape=(224,224,3))
x = layers.Rescaling(scale=1./255)(inputs)
x = layers.RandomCrop(160,160)(x)
x = layers.RandomFlip(mode="horizontal")(x)
base_model = tf.keras.applications.ResNet50(include_top=False, weights=None,pooling='avg')
x = base_model(x,training=False)
outputs = layers.Dense(1000, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="ResNet50_ImageNet")

if hvd.rank() == 0:
   print(model.summary())

opt = keras.optimizers.Adam(learning_rate=0.1*hvd.size())
# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt)

model.compile(optimizer=opt, loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'],
              experimental_run_tf_function = False)

callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=5, min_lr=0.00001),
             keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10),
                 # Horovod: broadcast initial variable states from rank 0 to all other processes.
                 # ensure consistent initialization of all workers when training is started with
                 # random weights or restored from a checkpoint.
                 hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                 # Horovod: average metrics among workers at the end of every epoch.
                 hvd.callbacks.MetricAverageCallback(),
                 # Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during the first three epochs.
                 hvd.callbacks.LearningRateWarmupCallback(0.1*hvd.size(), warmup_epochs=3, verbose=1)]

verbose = 2 if hvd.rank() ==0 else 0

start = time.time()
history = model.fit(train_dataset,epochs=epochs,validation_data=validation_dataset,callbacks=callbacks, verbose=verbose)
end = time.time()

model_name = model.name

if hvd.rank() == 0:
    print('Total time: ', round((end - start),2),'(s)')
    fname = 'ImageNet-' + model_name + '-' +hvd.size() + 'GPUs-reuse.h5'
    print('Saving model to ', fname)
    model.save(fname)

print('All done for rank', hvd.rank())
