#!/usr/bin/env python
# coding: utf-8

#An example pseudo code of loading audio for training a model is as follows:
Import tensorflow as tf
Import tensorflow_io as tfio
from tensorflow.keras.preprocessing.text import tokenizer

Audio = tfio.audio.AudioIoTensor(r‘brooklyn.flac’)
print(Audio)
# Gives and audio tensor of the audio file with its shape, data type and sampling rate

#Add tokenizer and create sequences from words
tokenizer = Tokenizer(num_words=10000, lower=True).fit_on_texts(r’brooklyn.flac’)
sequences = tokenizer.texts_on_sequences(‘r’brooklyn.flac”)

vocab_siz = len(tokenizer.word_index) + 1 #create a vocab size
for i,s in enumerate(sequences): #iterate through all training and testing words
    for j in range(1,len(s)):
        in_s,out_s = s[:j], s[j]
        pad(convert_to_categorical(in_s))
        pad(convert_to_categorical(out_s))
        X_list.append(in_s)
        Y_list.append(out_s)
Model = Model() #Develop model
Model.add(Input(shape=max_string_length, vocab_siz)) #add input layer
Model.add(LSTM(32))  #add LSTM and more layers as required

