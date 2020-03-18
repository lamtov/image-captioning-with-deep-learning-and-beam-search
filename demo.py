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
    import tensorflow as tf
    path=os.path.abspath("")
    import json
    import numpy as np
    import pickle as pickle
    
    def gen_vgg_feat():
        vgg_feat=[]
        for i in range(40)[1:38]:
            image='images/'+'anh'+str(i)+'.jpg'
            image_feat = extract_feautures.get_vgg_feat(image)
        
            vgg_feat.append(image_feat)
        with open('images/vgg_feat', 'wb') as fp:
            pickle.dump(vgg_feat, fp)
    
    #gen_vgg_feat()
    
    # =============================================================================
    # vgg_feat=[]
    # for i in range(5)[1:5]:
    #     with open ('images/vgg_feat'+str(i), 'rb') as fp:
    #         vgg_feat_i = pickle.load(fp)
    #         for j in vgg_feat_i:
    #             vgg_feat.append(j)
    # =============================================================================
    # =============================================================================
    # print(len(vgg_feat))
    # =============================================================================
    # =============================================================================
    # with open('images/vgg_feat', 'wb') as fp:
    #     pickle.dump(vgg_feat, fp)
    # =============================================================================
            
    with open ('images/vgg_feat', 'rb') as fp:
        vgg_feat = pickle.load(fp)
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
    def demoImage(i):
        img=mpimg.imread('images/anh'+str(i)+'.jpg')
        imgplot = plt.imshow(img)
        plt.axis("off")
        plt.show()
        captions=generate_new_caption(vgg_feat[i-1])
        for caption in captions:
            speak.Speak(utils.listToSentence(caption['sen'][1:]))
        
    for i in range(40)[1:37]:
        demoImage(i)
    
