import numpy as np
import pandas as pd
import os
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Model
import keras.backend.tensorflow_backend as K
from keras.models import load_model
import math
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

K.set_session(get_session(0.3))

class deConvolution:
    def __init__(self):
        self.server='/home/hejianjun/'
        self.project = 'firefox'
        self.valid = 'valid'
        self.dup='' #_without_dup
        self.csv_path = self.server+'valid-bug-report/dataset/'+self.project+'/'+self.project+self.dup+'.csv'
        self.w2v_path = self.server+'valid-bug-report/model/word2vec/'+self.project+self.dup+'/'
        self.save_path = self.server+'valid-bug-report/model/dl/'+self.project+self.dup+'/'
        self.key_gram_path = self.server+'valid-bug-report/key-gram/'+self.project+self.dup+'/'+self.valid+'/result.csv'
        self.EMBEDDING_LEN = 200
        self.MAX_SUM_SEQUENCE_LENGTH = 10
        self.MAX_DES_SEQUENCE_LENGTH = 50
        self.key_gram={}
    
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
            ids.append(dataFrame.at[i,'id'])
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
    
    def split_data_ordered(self, ids, texts, labels):
        texts_list = self.list_split(texts,10)
        labels_list = self.list_split(labels,10)
        ids_list = self.list_split(ids,10)
        
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
        return summary_list, description_list, labels_list, ids_list
    
    def lists_merge(self, lists):
        list_merge=[]
        for list in lists:
            list_merge = list_merge + list
        return list_merge
    
    def get_feature(self,model,summary,description,label,weight,bias,vector):
        index=0
        df=pd.DataFrame(columns=('index', 'probability'))
        for v, w in zip(vector, weight):
            probability =1/(1+ math.exp(-w*v))
            if probability>0.5:
                row={'index':index,'probability':probability}
                df=df.append(row,ignore_index=True)
            index = index+1
        return df
    
    def find_key_gram(self,model,df,word_index,id,summary,description,sum1_input_pool,sum2_input_pool,sum3_input_pool,des1_input_pool,des2_input_pool,des3_input_pool):
        setKey=set()
        for index,row in df.iterrows():
            quotient =int(int(row['index'])/128)
            if quotient == 0:
                input_pool=sum1_input_pool
                text=summary
                n=1
            elif quotient == 1:
                input_pool=sum2_input_pool
                text=summary
                n=2
            elif quotient == 2:
                input_pool=sum3_input_pool
                text=summary
                n=3
            elif quotient == 3:
                input_pool=des1_input_pool
                text=description
                n=2
            elif quotient == 4:
                input_pool=des2_input_pool
                text=description
                n=3
            elif quotient == 5:
                input_pool=des3_input_pool
                text=description
                n=4
            remainder=int(row['index'])%128
            start_index = np.argmax(input_pool[:,remainder])#no problem
            new_row={'feature_index':row['index'],'probability':row['probability'],'n':n,'start_index':start_index}
            list=text[start_index:start_index+n]
            new_list=[str(x) for x in list if x != 0]
            ###################
            '''
            if quotient == 0 or quotient == 1 or quotient == 2:
                for i in range(start_index,start_index+n):
                    list.append(self.get_key(word_index,summary[i])[0])
            else:
                for i in range(start_index,start_index+n):
                    list.append(self.get_key(word_index,description[i])[0])
            '''
            ###################
            key=' '.join(new_list)
            if key != '':
                setKey.add(key)
                value=new_row['probability']
                if key in self.key_gram:
                    self.key_gram[key][0]+=value
                    self.key_gram[key][1]+=1
                    self.key_gram[key][2]=self.key_gram[key][0]/self.key_gram[key][1]
                else:
                    self.key_gram[key]=[value,1,value,0,[]]

        for key in setKey:
            self.key_gram[key][3]+=1
            self.key_gram[key][4].append(id)
            #new_df=new_df.append(new_row,ignore_index=True)

        
    def get_key(self,dict, value):
        return [k for k, v in dict.items() if v == value]
    
    def get_all_key_gram(self,word_index,summary_list,description_list,label_list, ids_list):
        count = 1
        model = load_model(self.save_path+'model.h5')
        weight,bias = model.get_layer('fullyConn').get_weights()
        summary_test=[]
        description_test=[]
        label_test=[]
        summary_test=self.lists_merge(summary_list[0:9])
        description_test=self.lists_merge(description_list[0:9])
        label_test=self.lists_merge(label_list[0:9])
        ids_test=self.lists_merge(ids_list[0:9])
        summary_test = np.array(self.tokenizer(summary_test, word_index, self.MAX_SUM_SEQUENCE_LENGTH))
        description_test = np.array(self.tokenizer(description_test, word_index, self.MAX_DES_SEQUENCE_LENGTH))
        label_test=np.array(label_test)
        ids_test=np.array(ids_test)
        fullyConn_layer = Model(inputs=model.input,outputs=model.get_layer('fullyConn').input)
        vector = fullyConn_layer.predict([summary_test,description_test])
        sum1_pool_layer = Model(inputs=model.input,outputs=model.get_layer('sumPool1').input)
        sum1_input_pool = sum1_pool_layer.predict([summary_test,description_test])
        sum2_pool_layer = Model(inputs=model.input,outputs=model.get_layer('sumPool2').input)
        sum2_input_pool = sum2_pool_layer.predict([summary_test,description_test])
        sum3_pool_layer = Model(inputs=model.input,outputs=model.get_layer('sumPool3').input)
        sum3_input_pool = sum3_pool_layer.predict([summary_test,description_test])
        des1_pool_layer = Model(inputs=model.input,outputs=model.get_layer('desPool1').input)
        des1_input_pool = des1_pool_layer.predict([summary_test,description_test])
        des2_pool_layer = Model(inputs=model.input,outputs=model.get_layer('desPool2').input)
        des2_input_pool = des2_pool_layer.predict([summary_test,description_test])
        des3_pool_layer = Model(inputs=model.input,outputs=model.get_layer('desPool3').input)
        des3_input_pool = des3_pool_layer.predict([summary_test,description_test])
        for i in range(len(summary_test)):
            df=self.get_feature(model,summary_test[i],description_test[i],label_test[i],weight,bias,vector[i])
            self.find_key_gram(model,df,word_index,ids_test[i],summary_test[i],description_test[i],sum1_input_pool[i],sum2_input_pool[i],sum3_input_pool[i],des1_input_pool[i],des2_input_pool[i],des3_input_pool[i])
            print('process '+str(count)+'...')
            count=count+1
        print('Start constrcuting dataframe...........')
        result=pd.DataFrame([{'key-gram':key,'sum':sum,'count':count,'doc_count':doc_count,'probability':probability,'bug report':ids} for key,(sum,count,probability,doc_count,ids) in self.key_gram.items()],columns=['key-gram', 'sum','count','doc_count','probability','bug report'])
        print('Start sorting............')
        result=result.sort_values(by='probability' , ascending=False)
        print('Save.............')
        result.to_csv(self.key_gram_path,index=True)
        
if __name__ == '__main__':
    dc=deConvolution()
    with open(dc.w2v_path+'word_index.pkl','rb') as pkl_word_index:
        with open(dc.w2v_path+'embeddings_matrix.pkl','rb') as pkl_embeddings_matrix:
            word_index=pickle.load(pkl_word_index)
            embeddings_matrix=pickle.load(pkl_embeddings_matrix)
    ids, texts, labels = dc.load_file()
    summary_list, description_list, label_list, ids_list = dc.split_data_ordered(ids, texts, labels)
    dc.get_all_key_gram(word_index,summary_list,description_list,label_list, ids_list)
    print(dc.project+'...')
    #print(new_df)
    #new_df.to_csv('/home/hejianjun/valid-bug-report/result.csv',index=False)
            
    