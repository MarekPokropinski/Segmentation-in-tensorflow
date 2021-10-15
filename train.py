import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from model import unet_model

dataset, info = tfds.load('oxford_iiit_pet:3.*.*',
                          with_info=True, data_dir='D:/')

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE


def normalize(input_image, input_mask):
    """
    Normalize input image to range [0, 1] and input mask to values [0, 1, 2]
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def preprocess(datapoint):
    """
    Preprocessing function for oxford pet dataset
    """
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(
        datapoint['segmentation_mask'], (128, 128), method='nearest')
    input_mask = tf.cast(input_mask, tf.int32)

    return normalize(input_image, input_mask)


train_images = dataset['train'].map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE)


class ImageAugmentation(tf.keras.layers.Layer):
    def __init__(self, seed=1337):
        super().__init__()
        self.inputs_augmentations = [
            tf.keras.layers.RandomFlip(mode='horizontal', seed=seed),
            tf.keras.layers.RandomTranslation(
                (-0.2, 0.2), (-0.2, 0.2), seed=seed),
            tf.keras.layers.RandomRotation(0.2, seed=seed),
            tf.keras.layers.RandomZoom(0.2, seed=seed),
        ]
        self.labels_augmentations = [
            tf.keras.layers.RandomFlip(mode='horizontal', seed=seed),
            tf.keras.layers.RandomTranslation(
                (-0.2, 0.2), (-0.2, 0.2), seed=seed),
            tf.keras.layers.RandomRotation(0.2, seed=seed),
            tf.keras.layers.RandomZoom(0.2, seed=seed),
        ]

    def call(self, inputs, labels):
        for aug in self.inputs_augmentations:
            inputs = aug(inputs)

        for aug in self.labels_augmentations:
            labels = aug(labels)

        return inputs, labels



train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(ImageAugmentation())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_images.batch(BATCH_SIZE)


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# for images, masks in train_batches.take(2):
#   sample_image, sample_mask = images[0], masks[0]
#   display([sample_image, sample_mask])


class F1Score(tfa.metrics.F1Score):
    def __init__(self, num_classes: tfa.types.FloatTensorLike, average: str = None,  threshold=None,
                 name: str = 'f1_score',
                 dtype: tfa.types.AcceptableDTypes = None
                 ) -> None:
        super().__init__(num_classes, average, threshold, name, dtype)
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, shape=(-1, ))
        y_true = tf.one_hot(y_true, self.num_classes)
        y_pred = tf.reshape(y_pred, shape=(-1, self.num_classes))
        super().update_state(y_true, y_pred, sample_weight=None)


if __name__ == '__main__':
    model = unet_model(output_channels=3)
    f1_metric = F1Score(num_classes=3, average='macro')
    # f1_metric = tfa.metrics.F1Score(num_classes=3, average='macro')
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True), metrics=['accuracy', f1_metric])
    # tf.keras.utils.plot_model(model, show_shapes=True)

    VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE

    model_history = model.fit(train_batches, epochs=20,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_batches)

    model.save('model')
