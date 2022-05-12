# Training Encoder-Decoder model to represent word embeddings and finally
# save the trained model as 'model.h5'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

SOS_token = 0
EOS_token = 1
maxlen = 27

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            if(len(word.strip().lstrip()) > 0):
              self.addWord(word.strip().lstrip())


    def addWord(self, word):
        word = word.strip()
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def prepareData(lang1, lang2, file_):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, file_)
  #  src = pairs[:,0]
  #  src = [int(obs_str) for obs_str in src]
  #  src = torch.FloatTensor(src)
  #  print(src)
    for pair in pairs:
        if len(pair) < 2:
          continue
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    
    print(input_lang.name, input_lang.n_words, input_lang.word2index.keys())
    print(output_lang.name, output_lang.n_words, output_lang.word2index.keys())
    return input_lang, output_lang, pairs
    
def readLangs(lang1, lang2, file_):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(file_, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[s for s in l.split('|')] for l in lines]


    # Reverse pairs, make Lang instances
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def indexesFromSentence(lang, sentence):
    res = []
   # print(sentence)
    for word in sentence.split(' '):
      if len(word.strip().lstrip()) > 0:
        res.append(lang.word2index[word.strip().lstrip()])
    #    print(word)
        #return [lang.word2index[word.strip().lstrip()] for word in sentence.split(' ')]
    return res
    
def tensorFromSentence(lang, sentence):
	#print(sentence)
  indexes=[SOS_token]
  indexes.extend(indexesFromSentence(lang, sentence))
  indexes.append(EOS_token)
  return indexes

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
#    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
    
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X


# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y


# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(TransformerBlock(n_units, 2, 32))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model

input_lang, output_lang, pairs = prepareData('state', 'obs', 'penta.txt')
input_len = len(pairs)
word2vec = []
for i in range(input_len):
  word2vec.append(tensorsFromPair(pairs[i]))
src = [x[0] for x in word2vec]
tgt = [x[1] for x in word2vec]
split_index = int(np.floor(input_len*0.8))
split_index2 = int(np.floor(input_len*0.9))
print(src[:2], tgt[:2])
x_train, x_val, X = src[:split_index], src[split_index+1:split_index2], src[split_index2+1:split_index2+2]
print(x_train[:2])
y_train, y_val, y = tgt[:split_index], tgt[split_index+1:split_index2], tgt[split_index2+1:split_index2+2]
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen, padding='post', value=EOS_token)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen, padding='post', value=EOS_token)
X = keras.preprocessing.sequence.pad_sequences(X, maxlen=maxlen, padding='post', value=EOS_token)
X_ = src[split_index2+1:split_index2+2]
y_train = keras.preprocessing.sequence.pad_sequences(y_train, maxlen=maxlen, padding='post', value=EOS_token)
y_val = keras.preprocessing.sequence.pad_sequences(y_val, maxlen=maxlen, padding='post', value=EOS_token)
print(x_train[:2], y_train[:2])

#x_train = np.vstack([x_train, x_train])
x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)
X = np.array(X)
y = np.array(y)
output_dim = np.power(output_lang.n_words-1, 3)
# prepare training data

trainY = encode_output(y_train,  output_dim)
print(trainY.shape, 123)

# prepare validation data
testY = encode_output(y_val, output_dim)

# define model
model = define_model(input_lang.n_words,  output_dim, maxlen, maxlen, 32)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# summarize defined model
print(model.summary())

# fit model
filename = 'seq2seq.h5'
#checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(x_train, trainY, epochs=20, batch_size=64, validation_data=(x_val, testY))#, callbacks=[checkpoint], verbose=2)
