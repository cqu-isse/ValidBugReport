import pickle
import pandas as pd

class Translate:
    def __init__(self):
        self.server='/home/hejianjun/'
        self.project = 'firefox'
        self.valid='valid'
        self.dup='' #_without_dup
        self.min=1000
        self.max=100000
        self.csv_path = self.server+'valid-bug-report/dataset/'+self.project+'/'+self.project+self.dup+'.csv'
        self.w2v_path = self.server+'valid-bug-report/model/word2vec/'+self.project+self.dup+'/'
        self.key_index_gram_path = self.server+'valid-bug-report/key-gram/'+self.project+self.dup+'/'+self.valid+'/result.csv'
        self.key_gram_path = self.server+'valid-bug-report/key-gram/'+self.project+self.dup+'/'+self.valid+'/'+str(self.min)+'-'+str(self.max)+'.csv'
        
    
    def get_key(self,dict, value):
        return [k for k, v in dict.items() if v == value]
    
if __name__ == '__main__':
    trans=Translate()
    with open(trans.w2v_path+'word_index.pkl','rb') as pkl_word_index:
        word_index=pickle.load(pkl_word_index)
        
    #key_index_gram = pd.read_csv(trans.key_index_gram_path,nrows =5000)
    key_index_gram = pd.read_csv(trans.key_index_gram_path)
    key_index_gram=key_index_gram.sort_values(by='probability' , ascending=False)
    key_index_gram = key_index_gram.loc[(key_index_gram['doc_count']>trans.min) & (key_index_gram['doc_count']<trans.max)]
     
    key_gram=pd.DataFrame(columns=('key-gram','sum','count','doc_count','probability','bug reports'))
    
    for index,row in key_index_gram.iterrows():
        key=row['key-gram']
        value=row['probability']
        sum=row['sum']
        count=row['count']
        doc_count=row['doc_count']
        bug_reports=row['bug report']
        list=key.split(' ')
        new_list=[]
        for i in list:
            new_list.append(trans.get_key(word_index,int(i))[0])
        key=' '.join(new_list).strip()
        new_row={'key-gram':key,'sum':sum,'count':count,'doc_count':doc_count,'probability':value,'bug reports':bug_reports}
        key_gram=key_gram.append(new_row,ignore_index=True)
        
    
    key_gram.to_csv(trans.key_gram_path,index=False)
    
    

