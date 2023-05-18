# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

# Helper libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import portpicker
import multiprocessing

tf.keras.backend.clear_session()
print('Trying to resolve cluster')
cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=12345
)
print('Resolved cluster')
# use NCCL communication protocol
implementation = tf.distribute.experimental.CommunicationImplementation.NCCL
communication_options = tf.distribute.experimental.CommunicationOptions(implementation=implementation)

#declare distribution strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver,
                                                     communication_options=communication_options)
print('Number of replicas:', strategy.num_replicas_in_sync)

# get total number of workers
n_workers = int(os.environ['SLURM_NTASKS'])
print('Number of workers: ', n_workers)

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
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_dataset = train_dataset.with_options(options)
validation_dataset = validation_dataset.with_options(options)
train_dataset = train_dataset.batch(128*strategy.num_replicas_in_sync, drop_remainder=True)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(256*strategy.num_replicas_in_sync, drop_remainder=True)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


print('Steps per epoch: ', int((10000000/strategy.num_replicas_in_sync)/n_workers))

with strategy.scope():
    # the model
    inputs = keras.Input(shape=(224,224,3))
    x = layers.Rescaling(scale=1./255)(inputs)
    x = layers.RandomCrop(244,244)(x)
    x = layers.RandomFlip(mode="horizontal")(x)
    base_model = tf.keras.applications.ResNet50(include_top=False, weights=None,pooling='avg')
    x = base_model(x,training=False)
    #x = base_model(x)
    # x = layers.Flatten()(x)
    outputs = layers.Dense(1000, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="ResNet50_ImageNet_Multiworkers")
    model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])

callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=5, min_lr=0.00001),
             keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)]

history = model.fit(train_dataset,epochs=50,validation_data=validation_dataset,callbacks=callbacks, verbose=1)

model_name = model.name
fname = 'ImageNet-' + model_name + '-reuse.h5'
print('Saving model to ', fname)
model.save(fname)
