# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
#%pylab inline
import tensorflow as tf
import numpy as np
import os
import model as model
import utils as ultils

path=os.path.abspath("")

def train_modal():
    tf.reset_default_graph()
    config_model=ultils.Config()
    model_train=model.Model(config_model,0.5)
    init=tf.initialize_all_variables()
    loss_history = []
    save_file =path+ '/weight_model/'+model_train.config.model_name+'/train_model.ckpt'
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if(os.path.exists(save_file+'.index')):
            saver.restore(sess, save_file)
        else:
            sess.run(init)    
        for epoch in range(config_model.max_epochs):
            arrment=np.arange(config_model.total_train_img)
            np.random.shuffle(arrment)
            print('Epoch : '+str(epoch+1))
            total_loss=model_train.run_epoch(sess,model_train.config.batch_size, arrment)
            loss_history.extend(total_loss)
            
            print ('Mean loss: %.1f' % np.mean(total_loss))
            if epoch%1==0:
                save_f=path+ '/weight_model/'+model_train.config.model_name+'epoch_'+str(epoch)+'/train_model.ckpt'
                saver.save(sess, save_f)
                print('Model Saved.')
                        
def main():
    train_modal()
if __name__ == '__main__':
    main()

