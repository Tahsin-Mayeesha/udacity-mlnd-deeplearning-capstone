
# coding: utf-8

# In[1]:

import os
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing.image import ImageDataGenerator


# In[3]:

from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Lambda
from keras.layers.normalization import BatchNormalization


# In[16]:

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


# In[8]:

import numpy as np


# In[9]:

train_features = np.load('train_preprocesed.npy')
valid_features = np.load('valid_preprocessed.npy')


# In[10]:

train_dir = "new_train/"
valid_dir = "new_valid/"


# In[11]:

classes = os.listdir(train_dir)


# In[12]:

# Get the labels

train_labels = []
for c in classes:
    l = [c]*len(os.listdir(train_dir+c+'/'))
    train_labels.extend(l)
    


# In[25]:

len(train_labels)


# In[17]:

valid_labels = []

for c in classes:
    l = [c]*len(os.listdir(valid_dir+c+'/'))
    valid_labels.extend(l)


# In[18]:

onehot_train = to_categorical(LabelEncoder().fit_transform(train_labels))


# In[19]:

onehot_valid = to_categorical(LabelEncoder().fit_transform(valid_labels))


# In[20]:

vgg16_base = VGG16(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(150, 150,3))
# Note that the preprocessing of InceptionV3 is:
# (x / 255 - 0.5) x 2

print('Adding new layers...')
output = vgg16_base.get_layer(index = -1).output  
output = Flatten()(output)
# let's add a fully-connected layer
output = Dense(4096,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.5)(output)
output = Dense(512,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.5)(output)
# and a logistic layer -- let's say we have 200 classes
output = Dense(8, activation='softmax')(output)


vgg16_model = Model(vgg16_base.input, output)
#InceptionV3_model.summary()


# In[ ]:


for layer in vgg16_model.layers[:19]:
    layer.trainable = False

# In[21]:


vgg16_model.compile(optimizer="adam",loss="categorical_crossentropy",metrics =["accuracy"])


# In[35]:

train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)


val_datagen = ImageDataGenerator()



# In[38]:

callbacks = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')        
# autosave best Model
best_model_file = "./data_augmented_weights.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)


# In[39]:

history = vgg16_model.fit_generator(train_datagen.flow(train_features, onehot_train, batch_size=10), nb_epoch=5,
              samples_per_epoch = 3019,                     
              validation_data=val_datagen.flow(valid_features,onehot_valid,batch_size=10,shuffle=False),
                                    nb_val_samples=758,callbacks = [callbacks,best_model])


# In[34]:

#model.load_weights("batch_normalized_weights.h5")


# In[ ]:

# summarize history for accuracy
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['acc']); plt.plot(history.history['val_acc']);
plt.title('model accuracy'); plt.ylabel('accuracy');
plt.xlabel('epoch'); plt.legend(['train', 'valid'], loc='upper left');

# summarize history for loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss']); plt.plot(history.history['val_loss']);
plt.title('model loss'); plt.ylabel('loss');
plt.xlabel('epoch'); plt.legend(['train', 'valid'], loc='upper left');
plt.show()


# In[17]:

test_features = np.load("test_features.npy")


# In[18]:

test_preds = model.predict_proba(test_features, verbose=1)


# In[19]:

test_preds[0:5]


# In[21]:

submission1 = pd.DataFrame(test_preds, columns= os.listdir(train_dir))
test_files = os.listdir("test_stg1/test_stg1/")
submission1.insert(0, 'image', test_files)
submission1.head()


# In[27]:

clipped_preds = np.clip(test_preds,(1-0.82)/7,0.82)

submission2 = pd.DataFrame(clipped_preds, columns= os.listdir("train/train/"))
submission2.insert(0, 'image', test_files)
submission2.head()


# In[28]:

submission2.to_csv("batch_normalized.csv",index = False)


# In[ ]:



