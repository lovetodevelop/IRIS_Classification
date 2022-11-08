import tensorflow as tf # Machine learning
import pandas as pd # Data manipulation and analysis
import numpy as np # Numerical computation
import matplotlib.pyplot as plt # Plotting

categories = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
column_names = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm"]

# Load dataset
dataset = pd.read_csv("Iris.csv")

# Clean the data and create encoding 
dataset.dropna() 
dataset["Species"] = dataset["Species"].map(categories.index)

# Split the data into training and test sets
train = dataset.sample(frac=0.8, random_state=0)
test = dataset.drop(train.index)
print(len(train), 'training examples')
print(len(test), 'test examples')
train_X = (train.copy()).values
test_X = (test.copy()).values
train_y = (train.pop("Species")).values
test_y = (test.pop("Species")).values

# Normalization layer for data
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_X))
print(normalizer.mean.numpy())

# Build DNN model with 2 hidden layers
model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(categories))
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(
    train_X,
    train_y,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

def plot_loss(history):
  """
  plot_lost(history)
  Plots training history model.
  """
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

plot_loss(history) 

# Test model
test_loss, test_acc = model.evaluate(test_X,  test_y, verbose=2)
print('\nTest accuracy:', test_acc)
model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
