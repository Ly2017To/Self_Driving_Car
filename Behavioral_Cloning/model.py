### This code is developed based on the starter code provided in the class

#import libraries

import csv
import cv2
import numpy as np

#lines of .csv file
lines = [] 

with open('./collected_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# training data        
images = []
measurements = []

# steering_correction corresponds to image center,left,right 
steering_correction = [0,0.2,-0.2]

for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './collected_data/IMG/'+filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])+steering_correction[i]
        measurements.append(measurement)
        
   
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D

# normalize each image
# crop each image 
# apply the architecture invented by Nvidia https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
# this is a regression problem, so choose lose as mse
# choose adam optimizer 
# training 10 epochs

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2, shuffle=True)

model.save('model.h5')
