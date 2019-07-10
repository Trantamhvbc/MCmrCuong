import librosa
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt


# ham split2D_numpy()


def __splitstring(x):
    tmp = x.split()
    ret = []
    for i in range(86):
        ret.append(float(tmp[i]))
    return ret





def __load_file(myfile1, myfile2):
    lable = []
    retdata = []
    file = open(myfile1)
    s = file.readline()
    while len(s) > 0:
        x = []
        for i in range(128):
            #print(s)
            y = __splitstring(s)
            #print(y)
            x.append(y)
            s = file.readline()
        retdata.append(x)
    file = open(myfile2)
    s = "0"
    while len(s) > 0:
        s = file.readline()
        if len(s) > 0:
            lable.append([int(s)])
    return np.array( retdata), np.array( lable)


def __load_data():
    return __load_file("x_train.txt","y_train.txt"),__load_file("x_test.txt","y_test.txt")





batch_size = 128
num_classes = 2
epochs = 50

# input image dimensions
img_rows, img_cols = 128, 86

# the data, split between train and test sets

# 1s = 43 time
(x_train, y_train) ,(x_test, y_test) = __load_data()
#print(_x_tranning)
#print(_y_tranning)

print(x_train)
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#chuan hoa ve dang nhi phan
print("load data done")
x_train /= np.amax(x_train)
print('x_train shape:', x_train.shape)
print(x_train[0], 'train samples')
print(x_test.shape[0], 'test samples')

#convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data = (x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print('Test loss:', score[0])
print('Test accuracy:', score[1])

