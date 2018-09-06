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
import Train as former

class Multi_Trainer(former):

    seed = 7
    numpy.random.seed(seed)
    source_dir = None
    img_rows, img_cols = 42, 42
    input_shape = (img_rows, img_cols, 1)
    num_classes = None
    data = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    train = data.flow_from_directory(
        source_dir,
        target_size=(42, 42),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=800,
        subset="training")
    test = data.flow_from_directory(
        source_dir,
        target_size=(42, 42),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=800,
        subset="validation")

    def train_Classifajar(self):

        Multi_Trainer.source_dir = self.train_class
        Multi_Trainer.num_classes, dir_list = former.count_dir(
            Multi_Trainer.source_dir)
        model = self.large_model()
        train_map = Multi_Trainer.train.class_indices
        file = "./models_multi/labels_Classifajar.json"
        with open(file, 'w') as f:
            json.dump(train_map, f)
        print(train_map)
        model.fit_generator(
            Multi_Trainer.train,
            steps_per_epoch=100,
            epochs=10,
            validation_data=Multi_Trainer.test,
            validation_steps=50)
        model.save('./models_multi/model_Classifajar.h5')

        return print("Source dir: {}, num_classes: {}".format(
            Multi_Trainer.source_dir, Multi_Trainer.num_classes))

    def large_model(self):

        model = Sequential()
        model.add(Conv2D(44, (3, 3), input_shape=Multi_Trainer.input_shape,
                   activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(22, (3, 3), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(250, activation='relu'))
        model.add(Dense(Multi_Trainer.num_classes * 4, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(Multi_Trainer.num_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        return model

if __name__ == "__main__":
    avinas = Multi_Trainer()
    avinas.train_Classifajar()


