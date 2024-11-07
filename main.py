import os
import json
import tensorflow as tf
import mnist_setup  # Import dataset and model setup from mnist_setup.py

# Set batch size for each worker and calculate global batch size
per_worker_batch_size = 64
tf_config = json.loads(os.environ["TF_CONFIG"])
num_workers = len(tf_config["cluster"]["worker"])
global_batch_size = per_worker_batch_size * num_workers

# Initialize MultiWorkerMirroredStrategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Configure dataset options for sharding
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = (
    tf.data.experimental.AutoShardPolicy.AUTO
)


# Create distributed dataset with explicit options
def get_distributed_dataset():
    dataset = mnist_setup.mnist_dataset(global_batch_size)
    return dataset.with_options(options)


# Prepare the dataset within strategy scope
with strategy.scope():
    multi_worker_dataset = get_distributed_dataset()
    multi_worker_model = mnist_setup.build_and_compile_cnn_model()

# Set the number of steps per epoch
steps_per_epoch = 70  # Modify based on the dataset size

# Define callback for fault tolerance
backup_dir = "/tmp/backup"
callbacks = [tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir)]

# Train the model
multi_worker_model.fit(
    multi_worker_dataset, epochs=3, steps_per_epoch=steps_per_epoch, callbacks=callbacks
)