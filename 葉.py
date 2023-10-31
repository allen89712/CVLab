import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# plt.figure(figsize = (8, 8))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.title(str(y_train[i]), fontsize = 16, color = 'black', pad = 2)
#     plt.imshow(x_train[i], cmap = plt.cm.binary)
#     plt.xticks([])
#     plt.yticks([])

# plt.show()

val_images = x_test[:9000]
test_images = x_test[9000:]

val_images = val_images.astype("float32") / 255.0
val_images = np.reshape(val_images, (val_images.shape[0], 28, 28, 1))

test_images = test_images.astype("float32") / 255.0
test_images = np.reshape(test_images, (test_images.shape[0], 28, 28, 1))

train_images = x_train.astype("float32") / 255.0
train_images = np.reshape(train_images, (train_images.shape[0], 28, 28, 1))

factor = 0.17
val_noise_images = val_images + factor * np.random.normal(
    loc=0.0, scale=1.0, size=val_images.shape
)
test_noise_images = test_images + factor * np.random.normal(
    loc=0.0, scale=1.0, size=test_images.shape
)
train_noise_images = train_images + factor * np.random.normal(
    loc=0.0, scale=1.0, size=train_images.shape
)

val_noise_images = np.clip(val_noise_images, 0.0, 1.0)
train_noise_images = np.clip(train_noise_images, 0.0, 1.0)
test_noise_images = np.clip(test_noise_images, 0.0, 1.0)

# plt.figure(figsize = (8, 8))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.title(str(y_train[i]), fontsize = 16, color = 'black', pad = 2)
#     plt.imshow(train_noise_images[i].reshape(1, 28, 28)[0], cmap = plt.cm.binary)
#     plt.xticks([])
#     plt.yticks([])

# plt.show()

model = tf.keras.Sequential()

model.add(
    tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        activation="relu",
        padding="same",
        input_shape=(28, 28, 1),
    )
)
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation="relu", padding="same")
)
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

model.add(tf.keras.layers.UpSampling2D(size=2))
model.add(
    tf.keras.layers.Conv2DTranspose(
        filters=16, kernel_size=3, activation="relu", padding="same"
    )
)
model.add(tf.keras.layers.UpSampling2D(size=2))
model.add(
    tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, activation="relu", padding="same"
    )
)
model.add(tf.keras.layers.Activation("sigmoid"))

model.summary()

model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_noise_images,
    train_images,
    batch_size=256,
    epochs=10,
    validation_data=(val_noise_images, val_images),
)

plt.figure(figsize=(18, 18))
for i in range(10, 19):
    plt.subplot(9, 9, i)
    plt.imshow(test_images[i].reshape(1, 28, 28)[0], cmap=plt.cm.binary)
plt.show()


plt.figure(figsize=(18, 18))
for i in range(10, 19):
    plt.subplot(9, 9, i)
    plt.imshow(test_noise_images[i].reshape(1, 28, 28)[0], cmap=plt.cm.binary)
plt.show()


plt.figure(figsize=(18, 18))
for i in range(10, 19):
    plt.subplot(9, 9, i)
    plt.imshow(
        model.predict(test_noise_images[i]).reshape(1, 28, 28)[0],
        cmap=plt.cm.binary,
    )
plt.show()
