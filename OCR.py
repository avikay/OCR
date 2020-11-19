# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:51:46 2020

@author: Avinash
"""
import tensorflow as tf
import matplotlib.pyplot as plt


#Using a dataset of handwritten numbers 
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = tf.keras.utils.normalize(X_train, axis = 1)
X_test = tf.keras.utils.normalize(X_test, axis = 1)

#defining our model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

#Passing the vital parameters required for our model
model.compile(optimizer = 'adam', 
              loss= 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 3)

#Calculating loss and accuracy
val_loss, val_acc = model.evaluate(X_test, y_test)
#print(val_loss, val_acc)

#Saving our model
model.save('Optical_Character_Recognizer')

#Loading our model
new_model = tf.keras.models.load_model('Optical_Character_Recognizer')

#making predictions
predictions = new_model.predict([X_test])
print(predictions[0])

#Visualising original image to verify prediction
plt.imshow(X_test[0])
plt.show()

