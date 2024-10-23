import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from model import Model
from pgd_attack import LinfPGDAttack
import json
from datetime import datetime
from timeit import default_timer as timer
import os
import shutil

with open('config.json') as config_file:
    config = json.load(config_file)

# Set training parameters
tf.random.set_seed(config['random_seed'])
max_num_training_steps = config['max_num_training_steps']
batch_size = config['training_batch_size']

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28*28)  # Flatten images
x_test = x_test.reshape(-1, 28*28)

# Set up model
model = Model()

# Set up optimizer
optimizer = tf.keras.optimizers.Adam(1e-4)

# Set up adversary
attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

# Create a checkpoint directory if it doesn't exist
model_dir = config['model_dir']
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Main training loop
for step in range(max_num_training_steps):
    # Get a random batch
    indices = np.random.randint(0, x_train.shape[0], batch_size)
    x_batch, y_batch = x_train[indices], y_train[indices]

    # Compute adversarial perturbations
    start = timer()
    x_batch_adv = attack.perturb(x_batch, y_batch)
    end = timer()

    # Perform one step of training
    with tf.GradientTape() as tape:
        logits = model(x_batch_adv)
        loss_value = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch, logits=logits))
    
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if step % config['num_output_steps'] == 0:
        print(f"Step {step}: Loss = {loss_value.numpy()}")

    if step % config['num_checkpoint_steps'] == 0:
        model.save_weights(os.path.join(model_dir, f'checkpoint_{step}.h5'))
