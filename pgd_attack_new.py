import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from model import Model
import json

class LinfPGDAttack:
    def __init__(self, model, epsilon, k, a, random_start, loss_func):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_func = loss_func

    def perturb(self, x_nat, y):
        x = np.copy(x_nat)
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = np.clip(x, 0, 1)  # Ensure valid pixel range

        losses = []
        predictions = []
        
        for i in range(self.k):
            with tf.GradientTape() as tape:
                tape.watch(x)
                logits = self.model(x)
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
            
            grad = tape.gradient(loss, x)
            x = x + self.a * np.sign(grad)
            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)  # Stay within epsilon bounds
            x = np.clip(x, 0, 1)  # Ensure valid pixel range
            
            losses.append(loss.numpy())
            predictions.append(np.argmax(logits, axis=1))

        print(f"Losses: {losses}")
        print(f"Predictions: {predictions}")
        return x

if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)

    model = Model()
    attack = LinfPGDAttack(model, config['epsilon'], config['k'], config['a'], config['random_start'], config['loss_func'])

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0
    x_test = x_test.reshape(-1, 28*28)  # Flatten images

    # Choose a small batch for the attack
    x_adv = attack.perturb(x_test[:config['eval_batch_size']], y_test[:config['eval_batch_size']])

    np.save(config['store_adv_path'], x_adv)
    print(f"Adversarial examples saved at {config['store_adv_path']}")
