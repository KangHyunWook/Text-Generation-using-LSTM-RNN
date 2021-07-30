from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout    

import numpy as np
np.random.seed(2)

with open('Peter Pan.txt', 'r', encoding='utf-8') as f:
    corpus=f.read()
#extract uniquce characters from the corpus

corpus=[c for c in corpus.lower()]
charSet={c for c in corpus}

charSet=sorted(charSet)
ch2idx={}
idx2ch={}
#map characters to integers
for i, c in enumerate(charSet):
    ch2idx.update({c:i})
    idx2ch.update({i:c})
    
#prepare train data
trainX=[]
trainY=[]

window_size=100
for i in range(0, len(corpus)-window_size):
    X=[]
    for j in range(i, i+window_size, 1):
        #Convert characters to integers
        X.append(ch2idx[corpus[j]])
    Y=ch2idx[corpus[i+window_size]]
    trainX.append(X)
    trainY.append(Y)

dataX=trainX=np.array(trainX)
trainX=trainX.reshape(trainX.shape+(1,))
trainY=np.array(trainY)
#Noramlise input data
trainX=trainX/float(len(ch2idx))

n_chars=len(ch2idx)
def oneHotEncode(trainY):
    new_trainY=[]   
    for i in range(len(trainY)):
        Y=np.zeros(n_chars, dtype=np.uint8)
        Y[trainY[i]]=1
        new_trainY.append(Y)
    return np.array(new_trainY)
#One-hot encode labels 
trainY=oneHotEncode(trainY)

"""Pass the train data to the model"""
model=Sequential()
model.add(LSTM(256, input_shape=(window_size, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(n_chars))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(trainX, trainY, epochs=50, batch_size=64)

start=np.random.randint(0, len(trainX)-1)

pattern=trainX[start]
#upto last tutorial
import sys
print('=====generated text=====')
for i in range(1000):
    x=np.reshape(pattern, (1, len(pattern), 1))
    prediction=model.predict(x, verbose=0)
    index=np.argmax(prediction)
    res=idx2ch[index]
    sys.stdout.write(res)
    pattern[0:len(pattern)-1]=pattern[1:len(pattern)]
    pattern[len(pattern)-1]=index/float(n_chars-1)





















