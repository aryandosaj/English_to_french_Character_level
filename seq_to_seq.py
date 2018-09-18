data = open("fra.txt",'r',encoding = 'ascii',errors = 'ignore').readlines()
for i in range(len(data)):
    data[i] = data[i].split('\t')
    data[i][0] = list(data[i][0].strip('\t\n'))
    data[i][1] = list(data[i][1].strip('\t\n'))
    
txt = open("fra.txt",'r',encoding = 'ascii',errors = 'ignore').read()
vocab_size = len(list(set(txt)))
batch_size=64
epoch = 100
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback
import numpy as np
max_size = 10000
label = []
target = []
for line in data:
    label.append(line[0])
    target.append(line[1])
label = label[:10000]
target = target[:10000]
chars = sorted(list(set(txt)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
sen_len = 100
max_encoder_len = len(np.max(label))
max_decoder_len = len(np.max(target))
num_sen = len(target)
encoder_input = np.zeros((num_sen,sen_len,vocab_size),dtype='float32')
decoder_input = np.zeros((num_sen,sen_len,vocab_size),dtype='float32')
decoder_output = np.zeros((num_sen,sen_len,vocab_size),dtype='float32')
for i,(input_text,output_text) in enumerate(zip(label,target)):
    for t,c in enumerate(input_text):
        encoder_input[i,t,char_indices[c]]=1;
    for t,c in enumerate(output_text):
        decoder_input[i,t,char_indices[c]]=1;
        if t>0:
            decoder_output[i,t-1,char_indices[c]]=1

latent_dim = 256
from keras.models import Model
from keras.layers import Input, LSTM, Dense
encoder_model_inputs = Input(shape=(None, vocab_size))
encoder = LSTM(latent_dim, return_state=True)
encoder_model_outputs, state_h, state_c = encoder(encoder_model_inputs)
encoder_state = [state_h,state_c]

decoder_model_inputs = Input(shape=(None, vocab_size))
decoder = LSTM(latent_dim,return_sequences=True,return_state =True)
decoder_model_outputs, _, _=decoder(decoder_model_inputs)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_model_outputs = decoder_dense(decoder_model_outputs)



model = Model([encoder_model_inputs, decoder_model_inputs], decoder_model_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.summary()

print_callback = LambdaCallback(on_epoch_end=out)

model.fit([encoder_input, decoder_input], decoder_output,
          batch_size=batch_size,
          epochs=epoch,
          validation_split=0.2,
          callbacks=[print_callback])

def out(epoch,logs):
    test_sen = label[4000]
    test_sen2 = target[3999]
    x_test1=np.zeros((1,sen_len,len(chars)))
    x_test2=np.zeros((1,sen_len,len(chars)))

    for i,c in enumerate(test_sen):
        x_test1[0,i,char_indices[c]]=1
    for i,c in enumerate(test_sen2):
        x_test2[0,i,char_indices[c]]=1
    x_pred = model.predict([x_test1,x_test2],verbose=0)[0]
    pred_sen=[]
    print(x_pred)
    for l in x_pred:
        pred_sen.append(indices_char[np.argmax(l)])
    print(pred_sen)




