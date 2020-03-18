# -*- coding: utf-8 -*-
import material as materialwv
total_train_img=materialwv.total_train_img
total_test_img=materialwv.total_test_img
total_validate_img=materialwv.total_validate_img
dataset=materialwv.dataset
beam_size=1
class Config(object):
    img_dim=4096
    image_encoding_size=550
    hidden_dim=550
    word_encoding_size=materialwv.word_embedding_size
    word_enbedding_size=materialwv.word_embedding_size
    layers=max_len_sen=materialwv.max_len_sen
    vocab_size=materialwv.vocab_size
    batch_size=20*5 
    beam_size=beam_size
    max_len_gen=7
    seq_per_img=5
    max_epochs=13
    total_train_img=total_train_img
    model_name='model__'+ 'batch_size_'+str(batch_size)+'dataset_'+materialwv.dataset

def listToSentence(l):
    result=l[0]
    for i in l[1:]:
        result=result+' '+i
    return result