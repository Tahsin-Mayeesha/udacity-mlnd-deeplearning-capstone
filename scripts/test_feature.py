import os
import numpy as np
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

height = 150
width  = 150

def preprocess_image(path):
    img = image.load_img(path, target_size = (height, width))
    a = image.img_to_array(img)
    a = np.expand_dims(a, axis = 0)
    return preprocess_input(a)

test_path = ["test_stg1/test_stg1/"+name for name in os.listdir("test_stg1/test_stg1/")]

print("preprocessing images")
test_preprocessed_images = np.vstack(preprocess_image(fn) for fn in test_path)
np.save("test_preprocessed.npy",test_preprocessed_images)
print("preprocessing done and saved")


from keras.applications import VGG16
model = VGG16(weights="imagenet",include_top=False)
train_features = model.predict(test_preprocessed_images,batch_size=1,verbose=1)
np.save("test_features.npy",train_features)
