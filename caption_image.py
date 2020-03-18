# -*- coding: utf-8 -*-
%pylab inline
import extract_feautures as extract_feautures
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import model as model
import os
import win32com.client as wincl
import utils as utils
speak=wincl.Dispatch("SAPI.SpVoice")

path=os.path.abspath("")


def generate_new_caption( img_feat):
    tf.reset_default_graph()
    config=utils.Config()
    model_gen=model.Model(config,1)
    save_file =path+ '/weight_model/'+model_gen.config.model_name+'/train_model.ckpt'
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_file)
        
        captions=model_gen.generate_caption(sess,img_feat)[0:4]
        for caption in captions:
                    print(utils.listToSentence(caption['sen']), 'score: ', caption['score'])
    return captions  
def captionImage(image_path):
    img=mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.axis("off")
    plt.show()
    img_feat=extract_feautures.get_vgg_feat(image_path)
    captions=generate_new_caption(img_feat)
    for caption in captions:
        speak.Speak(utils.listToSentence(caption['sen'][1:]))
    

captionImage('images/ANHLOP.jpg')
