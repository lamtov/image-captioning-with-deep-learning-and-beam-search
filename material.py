
import scipy.io
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import json
import numpy as np

path=''
dataset = 'flickr8k'
#dataset = 'coco'
word_embedding_size=150
total_train_img=28000
total_validate_img=1000
total_test_img=1000
if dataset == 'flickr8k':
    total_train_img=6000
else:
    if dataset=='coco':
        word_embedding_size=100
        total_train_img=113000
        total_validate_img=5000
        total_test_img=5000

data_path=path+'data\\'+dataset+'\\'
result_path=path+'result_test\\'+dataset+'\\'
mat_feats = scipy.io.loadmat(data_path+'vgg_feats.mat')['feats'].T
print(mat_feats.shape)

#print("Done Load VGG_Feat: Matsize="+str(mat_feats.shape) )


stop_word=['his', 'because', 'shan', 'own', 'themselves', 'doesn', 'our', 'ourselves',
            'should', 'most',   'where', 'him',  'am','of',
           'wouldn', 'itself', 'your', 'll',  'their', 'ain', 'more', 'they', 
            'nor',  'weren',  'that',  'as', 'these', 'both', 'only',
           'than', 'here',  'so', 'herself', 'how', 's', 'myself', 't', 'has', 
           'her', 'further', 'himself', 'again', 'hers',  'very', 'just',
           'd',  'during', 'yourself', 'whom', 'which', 'or', 've', 'what', 
           'against', 're', 'aren', 'was', 'yours', 'for', 'm', 'don', 'didn', 'she', 'not',
           'y', 'been', 'its', 'mustn',  'ours',  'them', 'shouldn', 'you', 'few',
           'couldn', 'mightn', 'same', 'haven', 'ma', 'be', 'theirs', 'but', 'such', 'wasn', 'were',
           'those',  'did', 'too', 'about', 'who', 'isn', 'we', 'my','a', 
           'needn', 'i', 'when', 'then', 'once',  'will', 'won',
           'this', 'he', 'off',  'yourselves',  'it', 'had', 'why', 'hadn', 'hasn', 
           'can', 'until', 'no', 'being', 'do', 'any', 'if', 'o', 'now', 'me', 'does']

def check_AZ_word(word):
    if word=='STR' or word=='EOS':
        return True
    if word in stop_word:
        return False
    for i in word:
        if i>'z' or i<'a':
            return False
    return True
# print(check_AZ_word('adadsa2@dfe'))
data_sent = json.load(open(data_path+'dataset.json'))['images']


def getSenToken(imgid, sentid):
    sentence= data_sent[imgid]['sentences'][sentid]['tokens'][:total_train_img]
    sent=[]
    for word in sentence:
        if check_AZ_word(word)==True:
            sent.append(word)
    sent=sent[:48]
    sent.append('EOS')
    return sent
def getImageFeat(imgid):
    img_feats= mat_feats[imgid][:]
    return img_feats


#print("Done Load Sentences_Dataset MAX_LEN_SENTENCE= "+str(max_len_sen))

def createModel():
    sentences=[]
    for i in range(mat_feats.shape[0]):
        for j in range(5):
            sentences.append(getSenToken(i,j))        
    

    model = Word2Vec(sentences,size=word_embedding_size,window=10,min_count=5,workers=10)
    model.train(sentences, total_examples=len(sentences), epochs=1000)
    model.save('pretrained\\modelwv_'+dataset+'.bin')
#createModel()
    
##load model

word_vectors = KeyedVectors.load('pretrained\\modelwv_'+dataset+'.bin')
model_wordvec=word_vectors.wv


def index2Word(idx):
    return model_wordvec.index2word[idx]
def word2Index(wd):
    return model_wordvec.vocab[wd].index
def word2vec(wd):
    if wd==''or wd=='STR':
        return np.zeros((word_embedding_size,),dtype=float)
    return model_wordvec[wd]

vocab_size=len(model_wordvec.vocab)

def getMax_Len_Sen():
    max_len=0
    id1=id2=0
    for i in range(mat_feats.shape[0]):
        for j in range(5):
            if (len(getSenToken(i,j))+3)>max_len:
                max_len=len(getSenToken(i,j))+3
                id1=i
                id2=j
                
    max_len=max_len
    return max_len,id1,id2

max_len_sen,id1,id2=getMax_Len_Sen()

if max_len_sen>50:
    max_len_sen=50


def getSenCandidate(imgid, sentid):
    sen_token= data_sent[imgid]['sentences'][sentid]['tokens'][0:max_len_sen-3]
    sent=[]
    for word in sen_token:
        if word in model_wordvec.vocab:
            sent.append(word)
    return sent
def getTargetWord(imgid,sentid):
    sent= getSenCandidate(imgid, sentid)[:]
    sent.append('EOS')
    while(len(sent)<max_len_sen):
        sent.append('')
    return sent[:]
def getTargetVec(imgid,sentid):
    sentence =getTargetWord(imgid,sentid)[:]
    target_vec=[]
    for word in sentence:
        vec=0
        if word!='' and word!='STR':
           vec=word2Index(word)
        target_vec.append(vec)
    return target_vec[:]
def getSenVec(imgid,sentid):
    sentence= getSenCandidate(imgid, sentid)[:]
    sentence.insert(0,'STR')
    #sentence.append('EOS')
    while(len(sentence)<max_len_sen):
        sentence.append('')
    senvec=[]
    for word in sentence:
        senvec.append(word2vec(word))
    return senvec[:]

def getSenVecTemp(sent):
    sentence=sent[:]
    #if len(sentence)<max_len_sen:
        #sentence.append('EOS')
    while(len(sentence)<max_len_sen):
        sentence.append('')
    senvec=[]
    for word in sentence:
        senvec.append(word2vec(word))
    result=[]
    result.append(senvec)
    return result[:]
print(vocab_size,max_len_sen) 
def getImageLink(imgid):
    return data_sent[imgid]['filename']



