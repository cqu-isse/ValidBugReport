import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing import sequence
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from keras import layers
from keras.layers import Embedding
from keras.layers import Input, Dense, Conv1D
from keras.layers import Dropout, GlobalMaxPooling1D
from keras.models import Model
import keras
import keras.backend.tensorflow_backend as K
from keras.callbacks import EarlyStopping
import pickle

tf.set_random_seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
def get_session(gpu_fraction=0.4):
    """
    This function is to allocate GPU memory a specific fraction
    Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

K.set_session(get_session(0.4))

class DeepLearning:
    def __init__(self):
        self.server='/home/hejianjun/'
        self.project = 'netbeans'
        self.dup='' #_without_dup
        self.csv_path = self.server+'valid-bug-report/dataset/'+self.project+'/'+self.project+self.dup+'.csv'
        self.result_path = self.server+'valid-bug-report/result/'+self.project+'/result.csv'
        self.w2v_path = self.server+'valid-bug-report/model/word2vec/'+self.project+self.dup+'/'
        self.save_path = self.server+'valid-bug-report/model/dl/'+self.project+self.dup+'/'
        self.EMBEDDING_LEN = 200
        self.MAX_SUM_SEQUENCE_LENGTH = 10
        self.MAX_DES_SEQUENCE_LENGTH = 50
    
    def tokenizer(self, texts, word_index, MAX_SEQUENCE_LENGTH):
        data = []
        for sentence in texts:
            if sentence is np.nan:
                sentence = ' '
            new_txt = []
            for word in sentence.split(' '):
                try:
                    new_txt.append(word_index[word])  # 把句子中的 词语转化为index
                except:
                    new_txt.append(0)
            data.append(new_txt)
    
        texts = sequence.pad_sequences(data, maxlen = MAX_SEQUENCE_LENGTH)  # 使用kears的内置函数padding对齐句子,好处是输出numpy数组，不用自己转化了
        return texts
           
    def load_file(self):
        dataFrame = pd.read_csv(self.csv_path,encoding='utf-8')
        texts_summary = []   # 存储读取的 x
        texts_description = []
        labels = []  # 存储读取的y
        ids = []
        # 遍历 获取数据
        for i in range(len(dataFrame)):
            ids.append(dataFrame.at[i,'bug_id'])
            texts_summary.append(dataFrame.at[i,'summary']) # 每个元素为一句话“《机械设计基础》这本书的作者是谁？”
            texts_description.append(dataFrame.at[i,'description'])
            labels.append(dataFrame.at[i,'valid']) # 每个元素为一个int 代表类别 # [2, 6, ... 3] 的形式。减一为了 从0开始
            
        # 把类别从int 3 转换为(0,0,0,1,0,0)的形式
        #labels = to_categorical(np.asarray(labels)) # keras的处理方法，一定要学会# 此时为[[0. 0. 1. 0. 0. 0. 0.]....] 的形式
        texts=[]
        for text in zip(texts_summary,texts_description):
            texts.append(text)
        return ids, texts, labels  # 总文本，总标签
    
    
    def list_split(self, list, groups):
        len_list = int(len(list))+1
        size = int(len_list/groups)+1
        result=[]
        for i in range(0,len_list,size):
            item = list[i:i+size]
            result.append(item)
        return result
    
    def split_data_random(self, texts, labels):
        x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1)
        summary_train=[]
        summary_test=[]
        description_train=[]
        description_test=[]
        label_train=[]
        label_test=[]
        for text in x_train:
            summary_train.append(text[0])
            description_train.append(text[1])
        for text in x_test:
            summary_test.append(text[0])
            description_test.append(text[1])
        label_train = y_train
        label_test = y_test
        return summary_train, summary_test, description_train, description_test, label_train, label_test
    
    def split_data_ordered(self, ids, texts, labels):
        texts_list = self.list_split(texts,11)
        labels_list = self.list_split(labels,11)
        ids_list = self.list_split(ids,11)
        
        summary_list=[]
        description_list=[]
        for texts in texts_list:
            summary=[]
            description=[]
            for text in texts:
                summary.append(text[0])
                description.append(text[1])
            summary_list.append(summary)
            description_list.append(description)
        return summary_list, description_list, labels_list,ids_list
    
    def train_model(self,embeddings_matrix, summary_train,description_train, label_train):
        early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=2)
        
        summray_input = Input(shape=(self.MAX_SUM_SEQUENCE_LENGTH,),name='sumInput')
        
        description_input = Input(shape=(self.MAX_DES_SEQUENCE_LENGTH,),name='desInput')
        
        
        summary_embedding_layer = Embedding(input_dim=len(embeddings_matrix),  # 字典长度
                                output_dim = self.EMBEDDING_LEN,  # 词向量 长度（25）
                                weights=[embeddings_matrix],  # 重点：预训练的词向量系数
                                input_length=self.MAX_SUM_SEQUENCE_LENGTH,  # 每句话的 最大长度（必须padding） 10
                                trainable=False,  # 是否在 训练的过程中 更新词向量
                                name= 'summary_embedding'
                                )(summray_input)
        
        summary_embedding_layer = Dropout(0.5)(summary_embedding_layer)

        summary1 = Conv1D(128, kernel_size=1, strides=1, activation='relu', name='sumConv1')(summary_embedding_layer)
        summary1 = GlobalMaxPooling1D(name='sumPool1')(summary1)
        summary2 = Conv1D(128, kernel_size=2, strides=1, activation='relu', name='sumConv2')(summary_embedding_layer)
        summary2 = GlobalMaxPooling1D(name='sumPool2')(summary2)
        summary3 = Conv1D(128, kernel_size=3, strides=1, activation='relu', name='sumConv3')(summary_embedding_layer)
        summary3 = GlobalMaxPooling1D(name='sumPool3')(summary3)
        summary = layers.Concatenate(axis=-1,name='sumConcate1')([summary1,summary2])
        summary = layers.Concatenate(axis=-1,name='sumConcate2')([summary,summary3])
        
        description_embedding_layer = Embedding(input_dim=len(embeddings_matrix),  # 字典长度
                                output_dim = self.EMBEDDING_LEN,  # 词向量 长度（25）
                                weights=[embeddings_matrix],  # 重点：预训练的词向量系数
                                input_length=self.MAX_DES_SEQUENCE_LENGTH,  # 每句话的 最大长度（必须padding） 10
                                trainable=False,  # 是否在 训练的过程中 更新词向量
                                name= 'description_embedding'
                                )(description_input)
        
        description_embedding_layer = Dropout(0.5)(description_embedding_layer)
        

        description1 = Conv1D(128, kernel_size=2, strides=1, activation='relu', name='desConv1')(description_embedding_layer)
        description1 = GlobalMaxPooling1D(name='desPool1')(description1)
        description2 = Conv1D(128, kernel_size=3, strides=1, activation='relu', name='desConv2')(description_embedding_layer)
        description2 = GlobalMaxPooling1D(name='desPool2')(description2)
        description3 = Conv1D(128, kernel_size=4, strides=1, activation='relu', name='desConv3')(description_embedding_layer)
        description3 = GlobalMaxPooling1D(name='desPool3')(description3)
        description = layers.Concatenate(axis=-1,name='desConcate1')([description1,description2])
        description = layers.Concatenate(axis=-1,name='desConcate2')([description,description3])
      
        text = layers.Concatenate(axis=-1,name='concate')([summary,description])
        
        #text = Dropout(0.5)(text)
        #text = Dense(256, activation='relu')(text)
    
        #text = Dropout(0.3)(text)
        output = Dense(1, activation='sigmoid', name='fullyConn')(text)
        model = Model(inputs=[summray_input, description_input], outputs=output, name='model')
        #model.summary()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x=[summary_train, description_train], y=label_train, batch_size=32, epochs=300,validation_split=0.1,callbacks=[early_stopping])
        model.save(self.save_path+'model.h5') 
        
        return model
    
    def test_model(self, model, summary_test, description_test, label_test, id_test):
        result=model.predict([summary_test,description_test])
        tp=0
        tn=0
        fp=0
        fn=0
        predict=[]
        label=[]
        id=id_test
        for res,lab in zip(result,label_test):
            predict.append(res[0])
            label.append(int(lab))
            if res[0] >=0.50 and int(lab)==1:
                tp=tp+1
            if res[0] >=0.50 and int(lab)==0:
                fp=fp+1
            if res[0] <0.50 and int(lab)==1:
                fn=fn+1
            if res[0] <0.50 and int(lab)==0:
                tn=tn+1
        return tp, tn, fp, fn, predict, label, id #, pv, rv, pi, ri, f1v, f1i
    
    def lists_merge(self, lists):
        list_merge=[]
        for list in lists:
            list_merge = list_merge + list
        return list_merge
        
    def get_average(self, list):
        sum = 0
        for item in list:     
            sum += item  
        return sum/len(list)
    
    def get_sum(self, list):
        sum = 0
        for item in list:     
            sum += item  
        return sum
        
    def train(self,word_index,embeddings_matrix, summary_list,description_list, label_list, id_list):
        summary_train = self.lists_merge(summary_list[0:10])
        description_train = self.lists_merge(description_list[0:10])
        label_train = self.lists_merge(label_list[0:10])
        
        summary_test = summary_list[10]
        description_test = description_list[10]
        label_test = label_list[10]
        id_test = id_list[10]
    
        summary_train = np.array(self.tokenizer(summary_train, word_index, self.MAX_SUM_SEQUENCE_LENGTH))
        summary_test = np.array(self.tokenizer(summary_test, word_index, self.MAX_SUM_SEQUENCE_LENGTH))
        description_train = np.array(self.tokenizer(description_train, word_index, self.MAX_DES_SEQUENCE_LENGTH))
        description_test = np.array(self.tokenizer(description_test, word_index, self.MAX_DES_SEQUENCE_LENGTH))
        label_train=np.array(label_train)
        label_test=np.array(label_test)
        
        model= self.train_model(embeddings_matrix, summary_train, description_train, label_train)
        tp, tn, fp, fn, predict, label, id = self.test_model(model, summary_test, description_test, label_test, id_test)
        
        fpr, tpr, threshold = roc_curve(label, predict) ###计算真正率和假正率
        roc_auc = auc(fpr, tpr)
        print('-----------------------------')
        print('The results:')
        print('auc: '+str(roc_auc))
        print('tp: '+str(tp))  
        print('fp: '+str(fp))
        print('fn: '+str(fn))
        print('tn: '+str(tn))
        pv=tp/(tp+fp)
        print('pv: '+str(pv))
        rv=tp/(tp+fn)
        print('rv: '+str(rv))
        try:
            pi=tn/(tn+fn)
        except:
            pi=999
        print('pi: '+str(pi))
        ri=tn/(fp+tn)
        print('ri: '+str(ri))
        f1v=2*pv*rv/(pv+rv)
        print('f1v: '+str(f1v))
        f1i=2*pi*ri/(pi+ri)
        print('f1i: '+str(f1i))
        
        result=pd.DataFrame([{'bug_id':i,'predict':p,'label':l} for i,p,l in zip(id,predict,label)],columns=['bug_id', 'predict','label'])
        result.to_csv(self.result_path,index=False)
        
if __name__ == '__main__':
    dl=DeepLearning()
    with open(dl.w2v_path+'word_index.pkl','rb') as pkl_word_index:
        with open(dl.w2v_path+'embeddings_matrix.pkl','rb') as pkl_embeddings_matrix:
            word_index=pickle.load(pkl_word_index)
            embeddings_matrix=pickle.load(pkl_embeddings_matrix)
    ids, texts, labels = dl.load_file()
    summary_list, description_list, label_list, ids_list = dl.split_data_ordered(ids, texts, labels)
    dl.train(word_index,embeddings_matrix, summary_list, description_list, label_list,ids_list)
    