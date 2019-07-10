import librosa
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM, Flatten
from keras import backend as K

# def __cover_mel_spectrogram(my_file):
#     y, sr = librosa.load(my_file)
#     a = librosa.feature.melspectrogram(y=y, sr=sr)
#     y,x= np.shape(a)
#     #print(y )
#     #print(x)
#     return a
#
#
# # ham split2D_numpy()
# def split2D_numpy(a,vt):
#     y,x = np.shape(a)
#     ret1 = np.zeros((y,vt))
#     ret2 = np.zeros((y,x-vt))
#     for i in range(0,y):
#         for j in range(0,vt):
#             ret1[i][j]= a[i][j]
#             #print(a[i][j])
#     for i in range(0,y):
#         for j in range(vt,x):
#             ret2[i][j-vt] = a[i][j]
#     return np.array(ret1), np.array(ret2)
#
# def __splitstring(x):
#     ret1, ret2, ret3 = x.split()
#     return float(ret1) , float(ret2),ret3
#
#
# def read_file(name_file):
#     file = open(name_file,"r")
#     a = []
#     b = []
#     str = file.readline()
#     while(len(str) > 0):
#         x,y,z = __splitstring(str)
#         a.append([x,y])
#         b.append(z)
#         str = file.readline()
#     #print(a[1][0],a[1][1],b[1])
#     file.close()
#     return np.array(a),np.array(b)
#
#
#
#
# def __load_file(myfile1, myfile2):
#     arr, _lable_tranning = read_file(myfile1)
#     data = __cover_mel_spectrogram(myfile2)
#     i = 0
#     lable = []
#     retdata = []
#     while( i + 2 < arr[len(arr) - 1][1]):
#         X, Y = split2D_numpy(data, 86)
#         data = Y
#         for j in range( len(arr) ):
#             if i >= arr[j][0]:
#                 if i + 2 <= arr[j][1]:
#                     retdata.append(X)
#                     #print(X)
#                     if _lable_tranning[j] == 'no':
#                         lable.append([0])
#                     else:
#                         lable.append([1])
#                 elif j + 2 < len(arr):
#                     if i + 2 < arr[j + 2][0]:
#                         if _lable_tranning[j+1] == 'snore':
#                             retdata.append(X)
#                             #print(X)
#                             lable.append([1])
#         i = i + 2
#     return np.array( retdata), np.array( lable)

#
# def __load_data():
#     return __load_file("track1_6.txt","track1_6wav.wav"),__load_file("track1_1.txt","track1_1wav.wav")
'''
    buon cua cau
    
'''
batch_size = 128
num_classes = 2
epochs = 20
#
# def _load_file(myfile1,myfile2) :
#     dem = 0
#     file1 = open (myfile1,'r')
#     file2 = open (myfile2,'r')
#     ret_data = []
#     label = []
#     y = file2.readline()
#     while(len(y) > 0) :
#         label.append(y)
#         y = file2.readline()
#     file2.close()
#     x = file1.readline()
#     while (len(x) > 0):
#         if (dem % 128 == 0):
#             x1 = np.zeros((128, 86))
#             for i in range(0, 128):
#                 str = x.split()
#                 for j in range(len(str)):
#                     tt = float(str[j])
#                     x1[i][j] = tt
#                 x = file1.readline()
#         ret_data.append(np.array(x1))
#     file1.close()
#     return np.array(ret_data),np.array(label)

def __splitstring(x):
    tmp = x.split()
    ret = []
    for i in range(86):
        ret.append(float(tmp[i]))
    return ret





def _load_file(myfile1, myfile2):
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

def _load_data () :
    return _load_file("x_train.txt","y_train.txt"), _load_file("x_test.txt","y_test.txt")



# input image dimensions
img_rows, img_cols = 128, 86

# the data, split between train and test sets

# 1s = 43 time
(x_train, y_train) ,(x_test, y_test) = _load_data()
#print(_x_tranning)
#print(_y_tranning)
dem = 0
for i in range( len(  y_train )) :
    if y_train[i][0] == 1:
        dem = dem + 1
print(dem)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#chuan hoa ve dang nhi phan
x_train /= np.amax(x_train)
print('x_train shape:', x_train.shape)
print('y_train shape :',y_train.shape)
print(x_train[0], 'train samples')
print(x_test.shape[0], 'test samples')

#convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()

# IF you are running with a GPU, try out the CuDNNLSTM layer t ype instead (don't pass an activation, tanh is required)
model.add(LSTM(128, activation='relu',input_shape=(128, 86), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

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
print('Test loss:', score[0])
print('Test accuracy:', score[1])

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
