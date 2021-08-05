#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import numpy as np
import mnist
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from IPython.display import display, Image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from matplotlib import pyplot as plt


# # Training Data Set
#  Prepairing Image Sets with Labels as per the respective class

# In[18]:


train_batch = ImageDataGenerator().flow_from_directory('C:/Satvik/train_DataSet', 
                                                        target_size = (224, 224), 
                                                        classes = ['healthy_leaf', 'iron', 'nitrogen', 'phosphorus'],
                                                        class_mode = 'binary',
                                                        batch_size = 111,
                                                        )


train_images, train_labels = next(train_batch)

print(train_images.shape) 
print(train_labels.shape) 


# # Plot Function
# This function takes image data set as input and displays them with their respective class labels

# In[19]:


def plots(ims, figsize = (20, 20), rows = 11, interp = False, titles = None):
    defeciencyClass = ['Healthy_Leaf', 'Iron', 'Nitrogen', 'Phosphorus']
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0, 2, 3, 1))
        f = plt.figure(figsize = figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(defeciencyClass[titles[i]], fontsize = 16)
        plt.imshow(ims[i], interpolation = None if interp else 'none')


# # Size of the Training Data Set

# In[21]:


train_labels = train_labels.astype(int)

# Normalize the images.
train_images = (train_images / 255) - 0.5

print(train_images.shape) 


# # Building a Neural Network

# In[195]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()


# In[196]:


input_shape = (224, 224, 3)
num_classes = 4


# ## First Layer
# First layer consists of 32 filters with dimensions 3x3 with 'relu' activation function.

# In[197]:


model.add(Conv2D(32, kernel_size = (3, 3), strides = (1, 1),
                activation = 'relu',
                input_shape = input_shape))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))


# ## Second Layer
# Second layer consists of 64 filters with dimensions 3x3 with 'relu' activation function.

# In[198]:


model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))


# ## Output Layer
# Output layer has 2 nodes with 'softmax' activation function. Before applying 'softmax' activation, convolutional layers are flatten and 'relu' activation is applied with filter of 64 nodes.

# In[199]:


model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(num_classes, activation = 'softmax'))

print(model.summary())


# In[200]:


model.compile(
  optimizer = 'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)


# # Training of Model
# Here, the model trains for 10 epoches with batchsize of 9

# In[201]:


from tensorflow.keras.utils import to_categorical

model.fit(
  train_images,
  to_categorical(train_labels),
  batch_size = 11,
  epochs=5,
)


# # Saving the Model

# After the training of 'Model', all parameters (weights and biases) must be saved to use them further 

# In[202]:


model.save('cnn_2.h5')


# # Load the Model

# To use the model, for further predictions on 'Test Data', we need to load the pretrained model. Loading makes the model accesible to us.

# In[3]:


model = tf.keras.models.load_model("cnn_2.h5")


# ## Plot function for Test Data with Test Labels

# In[4]:


def plots(ims, figsize = (20, 20), rows = 11, interp = False, titles = None):
    defeciencyClass = ['Healthy_Leaf', 'Iron', 'Nitrogen', 'Phosphorus']
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0, 2, 3, 1))
        f = plt.figure(figsize = figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(defeciencyClass[titles[i]], fontsize = 16)
        plt.imshow(ims[i], interpolation = None if interp else 'none')


# # Test Data Set

# In[5]:


test_batch = ImageDataGenerator().flow_from_directory('C:/Satvik/test_DataSet_1', 
                                                      target_size = (224, 224), 
                                                      classes = ['test_healthy_leaf', 'test_iron', 'test_nitrogen', 'test_phosphorus'],
                                                      class_mode = 'binary',
                                                      batch_size = 12)

test_images, test_labels = next(test_batch)

test_images_normalized = (test_images / 255) - 0.5

print(test_images.shape)
print(test_labels.shape)


test_labels_1 = test_labels.astype(int)
plots(test_images, rows = 4, titles = test_labels_1)

print("Test Data Labels")


# ## Plot function for Test Data with Predicted Output

# In[6]:


def plots_1(ims, figsize = (20, 20), rows = 11, interp = False, titles = None):
    defeciencyClass = ['Healthy_Leaf', 'Iron-Sitrus Iron Nutrient Fertilizer', 'Nitrogen-Organic Nitrogen Rich Liquid Bacteria ', 'Phosphorus-Organic Phospho Rich Liquid Bacteria']
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0, 2, 3, 1))
        f = plt.figure(figsize = figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(defeciencyClass[titles[i]], fontsize = 16)
        plt.imshow(ims[i], interpolation = None if interp else 'none')


# # Predicted Output

# In[7]:


predictions = model.predict(test_images_normalized[:12])
predictions = np.argmax(predictions, axis = 1)
plots_1(test_images, rows = 5, titles = predictions)


# ## Test Accuracy

# In[9]:


import sklearn
from sklearn.metrics import accuracy_score
print("Test_accuracy = " + str(accuracy_score(test_labels, predictions)*100) + " %")


# In[90]:





# In[ ]:





# In[ ]:




