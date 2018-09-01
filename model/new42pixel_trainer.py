import numpy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import json
import os
#from keras import backend as K
#K.set_image_dim_ordering('tf')

seed = 7
numpy.random.seed(seed)
data = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train = data.flow_from_directory(
    "./../static/Emnist_dir/Own_classes/numbers/train",
    target_size=(42, 42),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=800,
    subset="training")
train_map = train.class_indices
print(train_map)

file = "./labels_own_numbers.json"
with open(file, 'w') as f:
    json.dump(train_map, f)

test = data.flow_from_directory(
    "./../static/Emnist_dir/Own_classes/numbers/train",
    target_size=(42, 42),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=800,
    subset="validation")
test_map = test.class_indices
print(test_map)
img_rows, img_cols = 42, 42
input_shape = (img_rows, img_cols, 1)
num_classes = len(next(os.walk("./../static/Emnist_dir/Own_classes/numbers/train"))[1])

def larger_model():

    model = Sequential()
    model.add(Conv2D(44, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(22, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(num_classes * 2, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

model = larger_model()
model.fit_generator(
        train,
        steps_per_epoch=100,
        epochs=10,
        validation_data=test,
        validation_steps=50)
model.save('kar_own_model_numbers.h5')



# Final evaluation of the model
scores = model.evaluate_generator(test)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
