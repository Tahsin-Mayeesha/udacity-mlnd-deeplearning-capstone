import numpy as np
from keras.applications import VGG16
model = VGG16(weights="imagenet",include_top=False)
train_images = np.load('train_preprocesed.npy')
train_features = model.predict(train_images,batch_size=1,verbose=1)
np.save("train_features.npy",train_features)
