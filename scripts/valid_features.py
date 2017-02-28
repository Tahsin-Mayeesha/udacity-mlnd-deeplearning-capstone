import numpy as np
from keras.applications import VGG16
model = VGG16(weights="imagenet",include_top=False)
valid_images = np.load('validation_images.npy')
valid_features = model.predict(valid_images,batch_size=1,verbose=1)
np.save("valid_features.npy",valid_features)
