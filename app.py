import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import os
from shutil import copyfile
import matplotlib.pyplot as plt
import json

labels = pd.read_csv('labels.csv')

labels_dict = {i:j for i,j in zip(labels['id'],labels['breed'])}
classes = set(labels_dict.values())
images = [f for f in os.listdir('train')]

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=0.3,
        height_shift_range=0.2,
        shear_range=0.3,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('training_images/afghan_hound/0379145880ad3978f9b80f0dc2c03fba.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='dog_breed', save_format='jpeg'):
    i += 1
    if i > 20:
        break

from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Activation, Flatten
from keras.layers import Dense
from keras.layers import Conv2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'training_images',
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'validation_images',
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')




model = Sequential()

# -----------------------------------------------------------------------------
# conv 1
model.add(Conv2D(16, (3,3), input_shape=(150, 150, 3)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

# max pool 1
model.add(MaxPooling2D(pool_size=(2,2),strides=2))

# -----------------------------------------------------------------------------
# # conv 2
model.add(Conv2D(32, (3,3)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

# max pool 2
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
# -----------------------------------------------------------------------------

# conv 3
model.add(Conv2D(48, (3,3)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
#model.add(Dropout(0.7))

# max pool 3
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
# -----------------------------------------------------------------------------

# # conv 4
model.add(Conv2D(64, (3,3)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
#model.add(Dropout(0.7))
# max pool 4
model.add(MaxPooling2D(pool_size=(2,2),strides=2))

# # conv 5
#model.add(Conv2D(128, (3,3)))
#model.add(BatchNormalization(axis=3))
#model.add(Activation('relu'))
#model.add(Dropout(0.7))
# max pool 4
#model.add(MaxPooling2D(pool_size=(2,2),strides=2))

# flatten
model.add(Flatten())

model.add(Dense(1024, activation='relu'))

# fc layer 2
model.add(Dense(512, activation='relu'))

# fc layer 3
model.add(Dense(256, activation='relu'))



# fc layer 4
model.add(Dense(120, activation='softmax'))


#metrics = accuracy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping_monitor=EarlyStopping(patience=6)

hist=model.fit_generator(
        training_set,
        steps_per_epoch=400,
        epochs=100,
        validation_data=test_set,
        validation_steps=2222,
        callbacks=[early_stopping_monitor])


#predictions_df = pd.DataFrame(predictions)
#predictions_df.columns = column_names
#predictions_df.insert(0,'id', test_set_ids)
#predictions_df.index = test_set_ids

#predictions_df.to_csv('finaloutput.csv',sep=",")

matplotlib.rcParams.update({'font.size': 14})


plt.plot(hist.history['accuracy'],label="Accuracy")
# plt.plot(hist.history['val_accuracy'], label="Validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.savefig('accuracy.png', dpi=300)

plt.clf()
plt.plot(hist.history['loss'],label="Training loss")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.savefig('loss.png', dpi=300)

with open('file.json','w') as f:
    json.dump(hist.history,f)


