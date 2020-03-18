# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os, time
import material as materialwv

tf.reset_default_graph()
path=os.path.abspath("")

class Model(object):
    def __init__(self,config,drop_prob):
        self.config=config
        self.drop_prob_output=drop_prob
        self.drop_prob_input=drop_prob
        self.beam_size=config.beam_size
        
        #placeholder
        self._batch_size_placeholder = tf.placeholder(tf.int32, [], name='batch_size')
        self._sent_placeholder= tf.placeholder(tf.float32,[None,self.config.layers,config.word_enbedding_size])
        self._img_placeholder=tf.placeholder(tf.float32,[None,self.config.img_dim])
        self._targets_placeholder=tf.placeholder(tf.int32,[None])
        self._dropout_input_placeholder = tf.placeholder(tf.float32, name='dropout_input_placeholder')
        self._dropout_output_placeholder = tf.placeholder(tf.float32, name='dropout_output_placeholder')
        print('self._sent_placeholder: ',self._sent_placeholder)
        print('self._img_placeholder: ',self._img_placeholder)
        print('self._targets_placeholder: ',self._targets_placeholder)
        
        
        
        
        with tf.variable_scope('CNN2HIDDEN'):
            W_i_c_1=tf.Variable(tf.random_normal([self.config.img_dim,self.config.hidden_dim]),'W_i_CNN2HIDDEN_CELL_STATE_1')
            b_i_c_1=tf.Variable(tf.random_normal([self.config.hidden_dim]),'b_i_CNN2HIDDEN_CELL_STATE_1')
            
            cell_state_1=tf.nn.relu(tf.nn.bias_add(tf.matmul(self._img_placeholder,W_i_c_1),b_i_c_1))
            
            
            W_i_h_1=tf.Variable(tf.random_normal([self.config.img_dim,self.config.hidden_dim]),'W_h_CNN2HIDDEN_HIDDEN_STATE_1')
            b_i_h_1=tf.Variable(tf.random_normal([self.config.hidden_dim]),'b_h_CNN2HIDDEN_HIDDEN_STATE_1')
            
            hidden_state_img_1=tf.nn.relu(tf.nn.bias_add(tf.matmul(self._img_placeholder,W_i_h_1),b_i_h_1))
            init_state_1=tf.nn.rnn_cell.LSTMStateTuple(cell_state_1,hidden_state_img_1)
            
            W_i_c_2=tf.zeros([self.config.img_dim,self.config.hidden_dim],dtype=tf.float32, name='W_i_CNN2HIDDEN_CELL_STATE_2')
            b_i_c_2=tf.Variable(tf.random_normal([self.config.hidden_dim]),'b_i_CNN2HIDDEN_CELL_STATE_2')
            
            cell_state_2=tf.nn.relu(tf.nn.bias_add(tf.matmul(self._img_placeholder,W_i_c_2),b_i_c_2))
           
            
            W_i_h_2=tf.zeros([self.config.img_dim,self.config.hidden_dim],dtype=tf.float32, name='W_h_CNN2HIDDEN_HIDDEN_STATE_2')
            b_i_h_2=tf.Variable(tf.random_normal([self.config.hidden_dim]),'b_h_CNN2HIDDEN_HIDDEN_STATE_2')
            
            hidden_state_img_2=tf.nn.relu(tf.nn.bias_add(tf.matmul(self._img_placeholder,W_i_h_2),b_i_h_2))
            init_state_2=tf.nn.rnn_cell.LSTMStateTuple(cell_state_2,hidden_state_img_2)
                                
            init_state=tuple([init_state_1,init_state_2])
            #print('init_state: ',init_state)
            
            
            
            #CREATE LSTM_CELL WIHT DROP_IN DROP OUT
            lstm_cell_1=tf.contrib.rnn.LSTMCell(self.config.hidden_dim, forget_bias=1,state_is_tuple=True, activation=tf.nn.tanh)#2 THAY TANH BANG SIGMOID
            lstm_dropout_1=tf.contrib.rnn.DropoutWrapper(lstm_cell_1,input_keep_prob=self._dropout_input_placeholder,output_keep_prob=self._dropout_output_placeholder)
            lstm_cell_2=tf.contrib.rnn.LSTMCell(self.config.hidden_dim, forget_bias=1,state_is_tuple=True, activation=tf.nn.tanh)#2 THAY TANH BANG SIGMOID
            lstm_dropout_2=tf.contrib.rnn.DropoutWrapper(lstm_cell_2,input_keep_prob=self._dropout_input_placeholder,output_keep_prob=self._dropout_output_placeholder)
           
            stacked_lstm=tf.contrib.rnn.MultiRNNCell([lstm_dropout_1,lstm_dropout_2], state_is_tuple=True)

         
            outputs,final_state=tf.nn.dynamic_rnn(stacked_lstm,self._sent_placeholder,initial_state=init_state, scope='LSTM')
            
            
            output=tf.reshape(outputs,[-1,self.config.hidden_dim])
            self._final_state=final_state
           
            print ('Output:', output)
            
        #Softmax layer
        with tf.variable_scope('sotfmax'):
            softmax_w=tf.Variable(tf.random_normal([self.config.hidden_dim,self.config.vocab_size]),'W_Softmax')
            softmax_b=tf.Variable(tf.random_normal([self.config.vocab_size]),'b_Softmax')
            logits=tf.nn.bias_add( tf.matmul(output,softmax_w),softmax_b)
            print('Logits: ',logits)
        self.logits=logits
        self._predictions=predictions=tf.argmax(logits,1) 
        self._softmax_logits=tf.nn.softmax(logits)
        print('Predictions: ', predictions)
        
# =============================================================================
#         #Minimize Loss
# =============================================================================
        
        with tf.variable_scope('loss'):
            logits_seq=tf.reshape(logits,[-1,self.config.layers,self.config.vocab_size])
            targets_seq=tf.reshape(self._targets_placeholder,[-1,self.config.layers])
            self.all_loss=all_loss=tf.contrib.seq2seq.sequence_loss(logits_seq,targets_seq,
                                                                    tf.ones([self._batch_size_placeholder, self.config.layers], dtype=tf.float32),average_across_timesteps=False,average_across_batch=True)
            self.loss=loss=tf.reduce_sum(all_loss)
            print('ALL Loss: ',all_loss)
        with tf.variable_scope('optimizer'):
# =============================================================================
#             optimizer=tf.train.RMSPropOptimizer( learning_rate=4e-4,
#                                                 decay=0.999,
#                                                 momentum=0.9,
#                                                 epsilon=1e-08,
#                                                 use_locking=False,
#                                                 centered=False,)
# =============================================================================
            
            optimizer=tf.train.AdamOptimizer( learning_rate=4e-4,
                                                beta1=0.8,
                                                beta2=0.999,
                                                epsilon=1e-08,
                                                use_locking=False
                                                )
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients = [
                None if gradient is None else tf.clip_by_norm(gradient, 5.0)
                for gradient in gradients]
            optimize = optimizer.apply_gradients(zip(gradients, variables))
            self.train_op=optimize
    def get_dataset(self,index,batch_size,arrment):
        img_placeholder=[]
        sent_placeholder=[]        
        targets_placeholder=[]
        batch=batch_size//5
        for j in range(5):
            for i in range(batch):
                idx=arrment[index*batch+i]
               
                img_feats=materialwv.getImageFeat(idx)
            
                img_placeholder.append(img_feats)
                sent_placeholder.append(materialwv.getSenVec(idx,j))
                targets_vec=materialwv.getTargetVec(idx,j)
                for k in targets_vec:
                    targets_placeholder.append(k)
       
        return {'img_placeholder':np.array(img_placeholder),'sent_placeholder':np.array(sent_placeholder),'targets_placeholder':np.array(targets_placeholder)}
            
            
    def run_epoch(self,session,batch_size, arrment):
        total_step=self.config.total_train_img//(self.config.batch_size//5)
        total_loss=[]
       
        
        for step in range(total_step):
            batch_data_set=self.get_dataset(step,batch_size,arrment)
            img_placeholder=batch_data_set['img_placeholder']
            sent_placeholder=batch_data_set['sent_placeholder']
            targets_placeholder=batch_data_set['targets_placeholder']
             
            feed={self._batch_size_placeholder:batch_size,
                  self._sent_placeholder:sent_placeholder,
                  self._img_placeholder:img_placeholder,
                  self._targets_placeholder:targets_placeholder,
                  self._dropout_input_placeholder:self.drop_prob_input,
                  self._dropout_output_placeholder:self.drop_prob_output
                  }
            loss,_=session.run([self.loss,self.train_op], feed_dict=feed)
            total_loss.append(loss)
            if(step%50==0):
                print("STEP : "+ str(step) +" LOSS = " + str(np.mean(total_loss)))
        return total_loss
    def runone(self,session,i,j,sentt):
        sent_placeholder=[]
        if sentt=='STR':
            sent=materialwv.getSenVec(i,j)
            sent_placeholder.append(sent)
        else :
            sent_placeholder=materialwv.getSenVecTemp(sentt)
        
        
        img=materialwv.getImageFeat(i)
        _img_placeholder=[]
        _img_placeholder.append(img)
        targets_placeholder=materialwv.getTargetVec(i,j)
       
        feed={self._batch_size_placeholder:1,
                  self._sent_placeholder:np.array(sent_placeholder),
                  self._img_placeholder:np.array(_img_placeholder),
                  self._targets_placeholder:np.array(targets_placeholder),
                  self._dropout_input_placeholder:self.drop_prob_input,
                  self._dropout_output_placeholder:self.drop_prob_output
                  }
        _predictions,_softmax_logits, loss,_=session.run([self._predictions,self._softmax_logits,self.loss,self.train_op], feed_dict=feed)
        print(materialwv.getSenCandidate(i,j))
        print('loss=',loss)
        predicsent=[]
        for n in _predictions:
            predicsent.append(materialwv.index2Word(n))
        print(predicsent)
        
        for n in range(3):
            
            maxxx=np.argsort(np.array(_softmax_logits[n]))[-5:]
            for kn in maxxx:
                print(materialwv.index2Word(kn), _softmax_logits[n][kn])
    def generate_caption(self,session,img_feauture):
        img=img_feauture[:]
        img_template=[]
        img_template.append(img)
        sent_start=[]
        sent_start.append('STR')
        endSearch=False
        cur_sents_input=[]
        cur_sents_input.append({'sen': sent_start,'score':1})
        step_search=0
        while endSearch==False:
            next_sents_input=[]
            step_search=step_search+1
            endSearch=True
            for sent_input in cur_sents_input:
                sent_pred=sent_input['sen']
                sent_score=sent_input['score']
                if len(sent_pred)<self.config.max_len_sen-1 and sent_pred[-1]!='EOS':
                    id_next_pred=len(sent_pred)-1
                    sent_template=materialwv.getSenVecTemp(sent_pred)
                    endSearch=False
                    softmax_logits=session.run(self._softmax_logits, feed_dict={
                            self._batch_size_placeholder:1,
                            self._sent_placeholder:np.array(sent_template), 
                            self._img_placeholder:np.array(img_template),
                            self._dropout_input_placeholder:self.drop_prob_input,
                            self._dropout_output_placeholder:self.drop_prob_output
                            })
                    predic_next=[]
                    if self.beam_size==1:
                        predic_next=np.argsort(np.array(softmax_logits[id_next_pred]))[-2:]
                    else :
                        predic_next=np.argsort(np.array(softmax_logits[id_next_pred]))[-1*(self.beam_size):]
                    for i in predic_next:
                        next_word=materialwv.index2Word(i)
                        if next_word != 'STR':
                            if step_search>=self.config.max_len_gen or next_word != 'EOS':
                                if (next_word=='EOS' and sent_score >= cur_sents_input[0]['score']) or next_word != 'EOS':
                                    next_score=softmax_logits[id_next_pred][i]*sent_score
                                    next_sent=sent_pred[:]
                                    next_sent.append(next_word)
                                    next_sents_input.append({'sen':next_sent,'score':next_score})
                
                                
                else:
                    next_sents_input.append({'sen':sent_pred,'score':sent_score})
              
            cur_sents_input=  sorted(next_sents_input,key=lambda k:k['score'],reverse=True)[0:self.beam_size]
            if cur_sents_input[0]['sen'][-1] !='EOS':
                cur_sent_forget=[]
                for senten in cur_sents_input:
                    if senten['sen'][-1] !='EOS':
                        cur_sent_forget.append(senten)
                cur_sents_input=cur_sent_forget[:]
            
            else :
                endSearch=True
        return cur_sents_input
            