import librosa
import numpy as np
import  string as st
def __cover_mel_spectrogram(my_file):
    y, sr = librosa.load(my_file)
    a = librosa.feature.melspectrogram(y=y, sr=sr)
    y,x= np.shape(a)
    #print(y )
    #print(x)
    return a


# ham split2D_numpy()
def split2D_numpy(a,vt):
    y,x = np.shape(a)
    ret1 = np.zeros((y,vt))
    ret2 = np.zeros((y,x-vt))
    for i in range(0,y):
        for j in range(0,vt):
            ret1[i][j]= a[i][j]
            #print(a[i][j])
    for i in range(0,y):
        for j in range(vt,x):
            ret2[i][j-vt] = a[i][j]
    return np.array(ret1), np.array(ret2)

def __splitstring(x):
    ret1, ret2, ret3 = x.split()
    return float(ret1) , float(ret2),ret3


def read_file(name_file):
    file = open(name_file,"r")
    a = []
    b = []
    str = file.readline()
    while(len(str) > 0):
        x,y,z = __splitstring(str)
        a.append([x,y])
        b.append(z)
        str = file.readline()
    #print(a[1][0],a[1][1],b[1])
    file.close()
    return np.array(a),np.array(b)




def __load_file(myfile1, myfile2):
    lable = []
    retdata = []
    for z in range(len(myfile1)):
        arr, _lable_tranning = read_file(myfile1[z])
        data = __cover_mel_spectrogram(myfile2[z])
        i = 0
        while( i + 2 < arr[len(arr) - 1][1]):
            X, Y = split2D_numpy(data, 86)
            data = Y
            for j in range( len(arr) ):
                if i >= arr[j][0]:
                    if i + 2 <= arr[j][1]:

                        #print(X)
                        if _lable_tranning[j] == 'no' and len(_lable_tranning) < 400:
                            retdata.append(X)
                            lable.append(0)
                        else:
                            retdata.append(X)
                            lable.append(1)
                    elif j + 2 < len(arr):
                        if i + 2 < arr[j + 2][0]:
                            if _lable_tranning[j+1] == 'snore':
                                retdata.append(X)
                                #print(X)
                                lable.append(1)
            i = i + 2
    for z in range(len(myfile1)):
        arr, _lable_tranning = read_file(myfile1[z])
        data = __cover_mel_spectrogram(myfile2[z])
        tmp,data = split2D_numpy(data,43)
        L1,L2  = np.shape(data)
        L2 /= 43
        i = 1
        while i + 2 < L2 :
            X, Y = split2D_numpy(data, 86)
            data = Y
            for j in range(len(arr)):
                if i >= arr[j][0]:
                    if i + 2 <= arr[j][1]:
                        # print(X)
                        if _lable_tranning[j] == 'snore':
                            retdata.append(X)
                            lable.append(1)
                    elif j + 2 < len(arr):
                        if i + 2 < arr[j + 2][0]:
                            if _lable_tranning[j + 1] == 'snore':
                                retdata.append(X)
                                lable.append(1)
            i = i + 2
    return np.array( retdata), np.array( lable)


def __load_data():
    myfile1 = []
    myfile2 = []
    myfiletest1 = []
    myfiletest2 = []
    myfiletest1.append("track1_1.txt")
    myfiletest1.append("track1_2.txt")
    myfile1.append("track1_3.txt")
    myfile1.append("track1_4.txt")
    myfile1.append("track1_5.txt")
    myfile1.append("track1_6.txt")
    myfile1.append("track1_7.txt")
    myfiletest2.append("track1_1wav.wav")
    myfiletest2.append("track1_2wav.wav")
    myfile2.append("track1_3wav.wav")
    myfile2.append("track1_4wav.wav")
    myfile2.append("track1_5wav.wav")
    myfile2.append("track1_6wav.wav")
    myfile2.append("track1_7wav.wav")
    return __load_file(myfile1,myfile2),__load_file(myfiletest1,myfiletest2)





batch_size = 128
num_classes = 2
epochs = 20

# input image dimensions
img_rows, img_cols = 128, 86

# the data, split between train and test sets

# 1s = 43 time
(x_train, y_train) ,(x_test, y_test) = __load_data()


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#chuan hoa ve dang nhi phan

file = open("x_train.txt",'w')
print(x_train[0][0][0])
for i in range(len(x_train)):
    for k in range(len(x_train[i])):
        for j in range( len(x_train[i][k])):
            file.write(str(x_train[i][k][j]))
            file.write(" ")
        file.write("\n")
file = open("x_test.txt",'w')
for i in range(len(x_test)):
    for k in range(len(x_test[i])):
        for j in range( len(x_test[i][k])):
            file.write(str(x_test[i][k][j]))
            file.write(" ")
        file.write("\n")


file = open("y_train.txt",'w')
for i in range(len(y_train)):
    file.write(str(y_train[i]))
    file.write("\n")
file = open("y_test.txt",'w')
for i in range(len(y_test)):
    file.write(str(y_test[i]))
    file.write("\n")
