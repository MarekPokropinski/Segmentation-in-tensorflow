import tensorflow as tf


def upsample_block(filters, kernel_size):
    initializer = tf.random_normal_initializer(0., 0.02)
    ct = tf.keras.layers.Conv2DTranspose(
        filters, kernel_size, strides=2, padding='same', use_bias=False, kernel_initializer=initializer)
    return tf.keras.Sequential([
        ct,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    ])


base_model = tf.keras.applications.MobileNetV2(
    input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(
    name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input,
                            outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    upsample_block(512, 3),  # 4x4 -> 8x8
    upsample_block(256, 3),  # 8x8 -> 16x16
    upsample_block(128, 3),  # 16x16 -> 32x32
    upsample_block(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(128, 128, 3))

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


