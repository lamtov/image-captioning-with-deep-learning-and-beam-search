%pylab inline
import tensorflow as tf
import numpy as np
import os, time
import material as materialwv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils as utils
import model as model
tf.reset_default_graph()
path=os.path.abspath("")
def test_modal(start,end):
    tf.reset_default_graph()
    config=utils.Config()
    model_test=model.Model(config,1)
    save_file =path+ '/weight_model/'+model_test.config.model_name+'/train_model.ckpt'
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_file)
        for i in range(200000)[start:end]:
            img=mpimg.imread(path+'/data/'+materialwv.dataset+'_Dataset/'+materialwv.getImageLink(i))
            imgplot = plt.imshow(img)
            plt.axis("off")
            plt.show()
            captions=model_test.generate_caption(sess,materialwv.getImageFeat(i))[0:4]
          
            for caption in captions:
                print(str(i),utils.listToSentence(caption['sen']), 'score: ', caption['score'])
          

def gen_var_caption():
    tf.reset_default_graph()
    config=utils.Config()
    model_test=model.Model(config,1)
    save_file =path+ '/weight_model/'+model_test.config.model_name+'/train_model.ckpt'
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_file)
        file=open(materialwv.data_path+'var\\'+"result_var.txt","w")
        start = time.time()
        for i in range(200000)[utils.total_train_img:utils.total_train_img+utils.total_validate_img]:
            captions=model_test.generate_caption(sess,materialwv.getImageFeat(i))[0]['sen']
            caption=utils.listToSentence(captions[1:len(captions)-1])
            file.write(caption+"\n")
            if i%100==0:
                print(i, "time",str(time.time()-start), caption)
                start=time.time()
        file.close()
def gen_test_caption():
    tf.reset_default_graph()
    config=utils.Config()
    model_test=model.Model(config,1)
    save_file =path+ '/weight_model/'+model_test.config.model_name+'/train_model.ckpt'
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_file)
        file=open(materialwv.result_path+"result_"+"beam"+str(utils.beam_size)+".txt","w")
        start = time.time()
        for i in range(200000)[utils.total_train_img+utils.total_validate_img:utils.total_train_img+utils.total_validate_img+utils.total_test_img]:
            captions=model_test.generate_caption(sess,materialwv.getImageFeat(i))[0]['sen']
            caption=utils.listToSentence(captions[1:len(captions)-1])
            file.write(caption+"\n")
            if i%100==0:
                print(i, "time",str(time.time()-start), caption)
                start=time.time()
        file.close()
test_modal(7000,7010)
#gen_test_caption()
#gen_var_caption()

