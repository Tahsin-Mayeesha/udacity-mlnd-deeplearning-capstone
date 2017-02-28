
# coding: utf-8

# In[25]:

import os
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:

from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Lambda
from keras.layers.normalization import BatchNormalization


# In[4]:

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


# In[5]:

import numpy as np


# In[6]:

train_features = np.load('train_features.npy')
valid_features = np.load('valid_features.npy')


# In[7]:

train_dir = "new_train/"
valid_dir = "new_valid/"


# In[8]:

classes = os.listdir(train_dir)


# In[9]:

# Get the labels

train_labels = []
for c in classes:
    l = [c]*len(os.listdir(train_dir+c+'/'))
    train_labels.extend(l)
    


# In[10]:

valid_labels = []

for c in classes:
    l = [c]*len(os.listdir(valid_dir+c+'/'))
    valid_labels.extend(l)


# In[11]:

onehot_train = to_categorical(LabelEncoder().fit_transform(train_labels))


# In[12]:

onehot_valid = to_categorical(LabelEncoder().fit_transform(valid_labels))


# In[ ]:




# In[32]:

model = Sequential()
model.add(Flatten(input_shape=train_features.shape[1:]))
model.add(Dense(4096, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(8, activation='softmax'))


# In[33]:


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics =["accuracy"])


# In[34]:

callbacks = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')        
# autosave best Model
best_model_file = "./batch_normalized_weights.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)


# In[ ]:

history = model.fit(train_features, onehot_train, batch_size=10, nb_epoch=10,
              validation_data=(valid_features,onehot_valid),shuffle=True,callbacks = [callbacks,best_model])


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


# In[20]:

test_features = np.load("test_features.npy")


# In[30]:

test_preds = model.predict_proba(test_features, verbose=1)


# In[31]:

test_preds[0:5]


# In[26]:

submission1 = pd.DataFrame(test_preds, columns= os.listdir(train_dir))
test_files = os.listdir("test_stg1/test_stg1/")
submission.insert(0, 'image', test_files)
submission.head()


# In[ ]:

clipped_preds = np.clip(preds,(1-0.82)/7,0.82)

submission2 = pd.DataFrame(clipped_preds, columns= os.listdir("train/train/"))
submission2.insert(0, 'image', test_files)
submission2.head()


# In[ ]:

submission2.to_csv("batch_normalized.csv",index = False)

