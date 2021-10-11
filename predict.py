from train import *
import numpy as np

model = tf.keras.models.load_model('model')

for image, mask in train_images:
  sample_image, sample_mask = image, mask
  [pred_mask] = model.predict(image[np.newaxis, ...])
  pred_mask = np.argmax(pred_mask, axis=-1)[..., np.newaxis]
  display([sample_image, sample_mask, pred_mask])