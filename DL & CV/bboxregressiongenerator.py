import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

class DataGenerator(Sequence):
    
    def __init__(self, df, X, y1, y2, batch_size, img_size, directory, shuffle=False):
        self.df = df.copy()
        self.X = X
        self.y1 = y1
        self.y2 = y2
        self.directory = directory
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.n = len(self.df)
        print(f"Number of records: {self.n}")
        print(f"Number of steps: {-(self.n // -self.batch_size)}")
        
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __len__(self):
        return -(self.n // -self.batch_size)
    
    def __getitem__(self,index):
    
        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size,:]
        X, y1, y2 = self.__get_data(batch)        
        return X, (y1,y2)
    
    def __get_data(self,batch):
        
        X,y1,y2 = [],[],[]
        for i, record in batch.iterrows():
            img = load_img(os.path.join(self.directory,record[self.X]),target_size=(self.img_size,self.img_size))
            img = img_to_array(img)
            img = img/255.
            X.append(img)
            coods = record[self.y1].values
            y1.append(coods)
            y2.append(record[self.y2])
        
        return np.array(X,dtype=np.float32),np.array(y1,dtype=np.float32),np.array(y2,dtype=np.float32)