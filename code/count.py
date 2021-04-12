import pandas as pd

server='/home/hejianjun/'
project='eclipse'
df = pd.read_csv(server+'valid-bug-report/statistics/'+project+'/'+project+'.csv')
print(df['has_attachment'].value_counts())
print(df['has_code'].value_counts())
print(df['has_comments'].value_counts())
print(df['has_environment'].value_counts())
print(df['has_error'].value_counts())
print(df['has_log'].value_counts())
print(df['has_other'].value_counts())
print(df['has_patch'].value_counts())
print(df['has_result'].value_counts())
print(df['has_screenshot'].value_counts())
print(df['has_stack_trace'].value_counts())
print(df['has_step_reproduce'].value_counts())
print(df['has_testcase'].value_counts())
