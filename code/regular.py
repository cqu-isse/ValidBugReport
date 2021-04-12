import keygram
import pandas as pd

server='/home/hejianjun/'
project='netbeans'

def keygram_extractor(text):
    keyg=keygram.Keygram(text)
    dict=keyg.extract()
    return dict

df = pd.read_csv(server+'valid-bug-report/dataset/'+project+'/'+project+'.csv')

df_stat = pd.DataFrame()


for index, row in df.iterrows():
    if row['valid']==1:
        file=open(server+'valid-bug-report/dataset/'+project+'/preprocess/'+str(row['bug_id'])+'.txt')#283782 743629
        text=file.read()
        file.close()
        keyg=keygram_extractor(text)
        keyg['bug_id']=row['bug_id']
        df_stat=df_stat.append(keyg,ignore_index=True)
        print('process '+str(row['bug_id'])+'...')
        
df_stat.to_csv(server+'valid-bug-report/statistics/'+project+'/'+project+'.csv',index=False)