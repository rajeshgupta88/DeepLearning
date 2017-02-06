
# coding: utf-8

# ### Referred from: http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/

# In[1]:

from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage


# In[2]:

# load data

(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[3]:

# creating a grid of 3*3 images

for i in range(0,9):
    pyplot.subplot(330+1+i)
    pyplot.imshow(toimage(X_train[i]))
    
pyplot.show()


# ## CNN Model for CIFAR-10

# In[4]:

import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


# In[5]:

# fix random seed for reproducibility
np.random.seed(seed=5)


# In[6]:

# Need to check why this step is giving error in the shape of the datasets
# normalize the input values from 0-255 to 0.0-1.0

X_train= X_train.astype('float32')
X_test = X_test.astype('float32')
X_train= X_train/255.0
X_test = X_test/255.0



# In[7]:

# The output variables are defined as a vector of integers from 0 to 1 for each of the 10 class.

# one hot encode outputs

y_train = np_utils.to_categorical(y_train)
y_test= np_utils.to_categorical(y_test)
num_classes= y_test.shape[1]


# In[8]:

print X_train.shape
print y_train.shape

print X_test.shape
print y_test.shape


# In[9]:

# y_train


# ## Simple CNN structure as a baseline

# 
# '''
# Two conv layer followed by max pooling and 
# then flatten out to fully connected layer to make predictions
# 
# 1. Each conv layer consist of a size of 3*3 maps with 32 feature , reLu function and weight constraint of max norm set to 3
# 2. Dropout set to 20%
# 3. Again Conv layer
# 4. Max pool layer with size 2*2
# 5. Flatten layer
# 6. Fully connected later with 512 units and a reLu
# 7. Dropout set to 50%
# 8. Fully connected output layer with 10 units and a softmax activation function
# 
# A logarithmic loss function is used with the stochastic gradient descent optimization algorithm configured 
# with a large momentum and weight decay start with a learning rate of 0.01
# '''
# 
# 

# In[10]:

# Create the model
model = Sequential()
model.add(Convolution2D(32,3,3, input_shape=(3,32,32),border_mode='same',activation='relu', W_constraint= maxnorm(3)))
model.add(Dropout(0.2))          
model.add(Convolution2D(32,3,3, activation='relu', border_mode='same', W_constraint= maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', W_constraint= maxnorm(3)))
model.add(Dropout(0.5))          
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
epochs = 25
lrate = 0.01
decay= lrate/epochs
          
sgd = SGD(lr = lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer= sgd, metrics = ['accuracy'])
print (model.summary())          


# In[11]:

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# ## Large Convolutional NN for CIFAR10
# 

# '''
# Network Architecture:
# 
# 1. Conv input layer, 32 feature maps with a 3*3 size kernel and reLu function
# 2. Dropout layer @ 20%
# 3. Conv layer, 32 feature maps with kernel size 3*3 and reLu
# 4. Max pool layer with size 2*2
# 5. Conv layer, 64 feature map with size 3*3 and reLu
# 6. Dropout layer at 20%
# 7. Conv layer, 64 feature map with size 3*3 and reLu
# 8. Max pool layer with size 2*2
# 9. Conv layer, 128 feature maps with 3*3 size and a reLu
# 10. Dropout layer at 20%
# 11. Conv layer, 128 feature maps with 3*3 size and a reLu
# 12. Max pool layer with size 2*2
# 13. Flatten Layer
# 14. Dropout layer @ 20%
# 15. Fully connected layer with 1024 units and a reLu
# 16. Dropout layer @ 20%
# 17. Fully connected layer with 512 units and a reLu
# 18. Dropout layer @ 20%
# 19. Full connected layer with 10 units and a softmax activation function
# 
# '''

# In[10]:

# create the model

model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(32,3,3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,3,3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(128,3,3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(128,3,3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))

model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))


# compile model

epochs = 25
lrate =0.01
decay= lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov= False)
model.compile(loss= 'categorical_crossentropy', optimizer=sgd, metrics= ['accuracy'])
print (model.summary())


# In[12]:

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:



