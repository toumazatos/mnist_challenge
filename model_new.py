import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        # Define the layers
        self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1024, activation='relu')
        self.fc2 = tf.keras.layers.Dense(10)

    def call(self, x):
        # Forward pass
        x = tf.reshape(x, [-1, 28, 28, 1])  # Reshape input into image format (batch, height, width, channels)
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        logits = self.fc2(x)  # No activation at the final layer, logits will be used for loss

        return logits

    def compute_loss(self, y_true, logits):
        # Loss function
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits))

    def compute_accuracy(self, y_true, logits):
        # Accuracy calculation
        predictions = tf.argmax(logits, axis=1)
        correct_prediction = tf.equal(predictions, y_true)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
