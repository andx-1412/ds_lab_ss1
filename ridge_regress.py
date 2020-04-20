import numpy as np
import os
import pandas as pd
class ridge_regression:
    def __int__(self):
        return
    def fit(self,  x_train , y_train , lamda):
        
      #  print( np.linalg.inv( np.transpose(x_train).dot(x_train) + lamda*np.identity(x_train.shape[1]) ).dot(np.transpose(x_train)).dot(y_train))
        return np.linalg.inv( np.transpose(x_train).dot(x_train) + lamda*np.identity(x_train.shape[1]) ).dot(np.transpose(x_train)).dot(y_train)
  

    def normalize_addone(self,x):
        x_max = np.array([[np.amax(x[:,i]) for i in range(x.shape[1])]]*x.shape[0])
        x_min = np.array([[np.amin(x[:,i]) for i in range(x.shape[1])]]*x.shape[0])
        #print(x_max)
        x = ( x - x_min)/(x_max - x_min)

        return np.column_stack((np.array([[1]] *(x.shape[0])),x))
    def predict(self,w,x):

        return x.dot(w)
        
    def loss(self, y_train, y_pre, w, lamda):
        
        return (1.0/y_train.shape[0])*np.sum((y_pre  - y_train)**2) + lamda*(np.sum(w*w)**(0.5))
    def fit_grad(self, x_train, y_train, lamda, learning_rate, batch_size, epoch = 100):
        w =np.array(np.random.normal() for i in range(x_train.shape[1]))
        last_loss = 10e+8
        batch_num = int(np.ceil( x_train.shape[0]/ batch_size ))
        for e in range(epoch):
            arra = np.array(range(x_train.shape[0]))
            np.random.shuffle(arra)
            new_x = x_train[arra]
            new_y = y_train[arra]
            
            for i in range(batch_range):

                x_batch =new_x[0+ i*batch_size: 0 + (i+1)*batch_size]
                y_batch = new_y[i*batch_size: (i+1)*batch_size]
                grad= x_batch.tranpose.dot(x_batch.dot(w) - y_batch ) + lamda*w
                w = w- learning_rate*grad
            new_loss = self.loss(self,y_train, self.predict(self, w, x_train),w )
            if(last_loss - new_loss <= 1e-5):
               break;
            last_loss = new_loss
        return w
    
          

                

        
        #for i in range(numb):

    

    def find_lamda(self, x_train, y_train):
        def range_scan_lamda(current_loss, best_lamda, attr, lamda_rand = range(50)):
         for lamda in lamda_rand:
            avg_loss = cross_validation(5, x_train, y_train, lamda)
            if(avg_loss < current_loss ):
                current_loss = avg_loss
                best_lamda = lamda
         return current_loss, best_lamda
        

        def cross_validation( numb,x_train,y_train, lamda):
            avg_loss =0
        
            row = np.array(range(x_train.shape[0]))
        
            val= np.array(np.split(row[:len(row) - len(row)%numb], numb,0 ))
            
            
            val[-1]=   np.append(val[-1],  row[len(row) - len(row)%numb:], 0)
            train = [[k for k in row if k not in val[i] ] for i in range(numb)]
            
            
            for i in range(numb):
                 valid_part = {'X' : x_train[val[i]], 'Y': x_train[val[i]]}
                 train_part = {'X' : x_train[train[i]], 'Y': y_train[train[i]]}
                 w = self.fit(train_part['X'], train_part['Y'], lamda )  
                 avg_loss+= self.loss(valid_part['Y'], self.predict(w,valid_part['Y']),w, lamda)
          #  print(avg_loss/numb)
            return avg_loss/numb
        current_loss, best_lamda = range_scan_lamda(100000000,0,1)
        lamda_range = [k/1000. for k in range(max(0,(best_lamda-1)*1000), (best_lamda+1)*1000,1)]
       # print(current_loss, best_lamda)
        current_loss, best_lamda = range_scan_lamda(current_loss, best_lamda,1, lamda_range)
        return best_lamda


ara = np.array([[1,2],[3,4], [5,6]]) 
#print(ara)
ara2 =np.array( np.split(ara,[1,2], 0 ) )
ara3=np.array( np.split(ara,2,1 ) )
#print(ara3[-1])
ara1=np.array( [[10],[11],[12]] )
#print(np.append(ara1, ara3[-1],0))
#print(ara1)
line_data= list()
with open("D:\death_rate.txt") as f:
    line = f.read().splitlines()
for i in range(len(line)):
  
    
    line_data.append(line[i].split())
    for k in range(0, len(line_data[i])) :
     # print(type(line_data[i][k]), line_data[i][k],i,k)
      line_data[i][k] = float(line_data[i][k])
line_data= np.array(line_data)
print(line_data)
x_train = line_data[0:50, 1: line_data.shape[1] -1]

y_train = line_data[0:50, line_data.shape[1] -1 : ]
x_test= line_data[50:, 1: line_data.shape[1]-1]
y_test = line_data[50:, line_data.shape[1]-1 :]
rd= ridge_regression()
x= rd.normalize_addone(x_train)

best_lamda = rd.find_lamda(x,y_train)
print(best_lamda)
w = rd.fit(x_train,y_train, best_lamda)
loss= rd.loss(y_test,rd.predict(w,x_test),w,best_lamda)
print(rd.predict(w,x_test))
print(loss)


