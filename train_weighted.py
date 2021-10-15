from train import *

class_weights = [0.7, 0.35, 2.0]

def add_sample_weights(image, label):
    class_weights_tensor = tf.constant(class_weights)
    class_weights_tensor = class_weights_tensor/tf.reduce_sum(class_weights_tensor)
    sample_weights = tf.gather(class_weights_tensor, indices=tf.cast(label, tf.int32))
    return image, label, sample_weights

if __name__ == '__main__':
    train_batches = train_batches.map(add_sample_weights)
    model = unet_model(output_channels=3)
    f1_metric = F1Score(num_classes=3, average='macro')
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True), metrics=['accuracy', f1_metric])

    VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE

    model_history = model.fit(train_batches, epochs=20,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_batches)

    model.save('weighted_model')
