import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images,train_labels), (test_images,test_labels) = data.load_data()

#giving names of the object classifying
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
              
#to reduce the data size of images by 255 pixels eg(234/255.0 or 211/255.0)
train_images = train_images/255.0
test_images = test_images/255.0

#to see print the image &  accuracy 
#plt.imshow(train_images[7], cmap=plt.cm.binary)
#plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"]
)

#model training times == epochs
model.fit(train_images,train_labels, epochs=8)

#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print("Tested Accuracy : ", test_acc)


#predicting the images objects
prediction = model.predict(test_images)

#i = no. of objects to predict
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
