from sklearn.model_selection import train_test_split
import os
import shutil
import pandas as pd



def new_distrib():
    y=pd.read_csv('DATA/X_target.csv')
    y['patient'] = y.image_filename.apply(lambda x :x.split('.')[0].split('_')[-1])
    sep_df = y.groupby('patient').min()
    train,test = train_test_split(sep_df,stratify=sep_df.class_number, test_size=0.1)
    y['set']= y.patient.apply(lambda x: 'train' if x in train.index else 'test')
    y.class_number = y.class_number.astype(str)
    print('Répartion : \nTrain:',sum(y.set=='train'),"Test:",sum(y.set=='test'),sep='\n')
    ###
    # Change Filename
    y.image_filename = y.apply(lambda x: f'DATA/Train/{x.class_number}/{x.image_filename}',axis=1)
    display(y[y.set=='train'].groupby('class_number').count())
    display(y[y.set=='test'].groupby('class_number').count())
    y.to_csv('splits.csv',index=False)
    
def builds_folders():
    y=pd.read_csv('DATA/X_target.csv') 
    train, test = train_test_split(y,test_size=0.2)
    try:
        os.mkdir(f'DATA/Train/')
        for x in frozenset(y.class_number):
            os.mkdir(f'DATA/Train/{x}')
    except:pass

    try:
        os.mkdir(f'DATA/Test/')
        for x in frozenset(y.class_number):
            os.mkdir(f'DATA/Test/{x}')
    except:pass

    for idx,elem in train.iterrows():
        curr = f'DATA/TrainingSetImagesDir/{elem.image_filename}'
        new = f'DATA/Train/{elem.class_number}/{elem.image_filename}'
        shutil.move(curr,new)
        
    for idx,elem in test.iterrows():
        curr = f'DATA/TrainingSetImagesDir/{elem.image_filename}'
        new = f'DATA/Test/{elem.class_number}/{elem.image_filename}'
        shutil.move(curr,new)
        
if __name__=="__main__":
    builds_folders()
