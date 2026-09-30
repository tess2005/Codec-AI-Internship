import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Load MNIST Dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalize and Reshape Data
X_train = X_train / 255.0
X_test = X_test / 255.0

X_train_cnn = X_train.reshape((-1, 28, 28, 1))
X_test_cnn = X_test.reshape((-1, 28, 28, 1))

# 3. Build Convolutional Neural Network
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 4. Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train Model
model.fit(X_train_cnn, y_train, epochs=5, validation_data=(X_test_cnn, y_test))

# 6. Evaluate Accuracy
test_loss, test_acc = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")

# 7. Save Sample Visualizations
predictions = model.predict(X_test_cnn[:5])
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f"Pred: {np.argmax(predictions[i])}\nTrue: {y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig("predictions_sample.png")
plt.show()