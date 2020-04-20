import os
import re
import nltk
import numpy as np
path = 'E:/20news-bydate.tar/'
sup_path = [path + namepath + "/" for namepath in os.listdir(path) if not os.path.isfile(path + namepath)]
train_path, test_path = [ sup_path[0], sup_path[1]] if 'train' in sup_path[0] else [sup_path[1], sup_path[0]]
topic = [name_path + '/' for name_path in os.listdir(train_path) if not os.path.isfile(train_path + name_path)]
stopword= 'E:/20news-bydate.tar/stopwords.txt'
f = open(stopword,'r')
stopwords = f.read()
#print(stopwords)
stem = nltk.PorterStemmer()
lis =['a','b','c']
lis2= [word for word in lis]
#print(lis2)

def get_file_from(path, sub_dir):
    data_file = []
    for sub_dir_id, name_sub_dir in enumerate(sub_dir):
        print(str(sub_dir_id))
        full_path = path + name_sub_dir
        list_file =[ (file_name, full_path+ file_name) for file_name in os.listdir(full_path) if os.path.isfile(full_path+ file_name)]
        list_file.sort()
        for file_name, file_path in list_file : 
            with open(file_path) as f:
                text= f.read().lower().split()
              #  print(text)
                text = [stem.stem(word) for word in re.split('\W+',str(text)) if not word in stopwords]
                content = ' '.join(text)
                assert len(content.splitlines()) ==1
                data_file.append(str(sub_dir_id) +'<fff>' +file_name+ '<fff>' +  content)
    #print(data_file)
    return data_file
#train_data = get_file_from(train_path, topic)
#f = open('E:/newtext.txt','w')
#f.write('\n'.join(train_data))
                

def idf(fre, times):
    return np.log10(times/fre)


def dic_idf(data_path):
    dic = open(data_path,'r')
    dic = dic.read()
    dic = dic.splitlines()
    
    dic_size = len(dic)
    tudien = dict()
   
    for line in dic:
        feature  = line.split('<fff>')
        words = [word for word in feature[-1].split() ]
        for word in words:
            if(tudien.get(word,False) == False):
                tudien[word]=1
            else: tudien[word] += 1
    dict_idf = [(word,idf(time, dic_size)) for word,time in zip(tudien.keys(), tudien.values()) if time > 10 and not word.isdigit()]
    def inside(val1):
        return -val1[1]
   # print(dic_idf)
    dict_idf.sort(key = inside )
    f = open('E:/dict_idf.txt','w')
    for i in range(len(dict_idf)):
        f.write(dict_idf[i][0] + '<fff>' + str(dict_idf[i][1]) + '\n')
    
        

def tf_idf(data_path, dict_idf_path):
    word_indict = open(dict_idf_path,'r').read().splitlines()
    dict_idf = dict()
    list_dict = list()
    for line in word_indict:
        here = line.split('<fff>')
        list_dict.append((here[0], here[1]))
   #print(list_dict)
    dict_idf = dict(list_dict)
    data = open(data_path,'r')
    #print(dict_idf)
    lines = data.read().splitlines()
    docs = list()
    term = list()
    data = list()
    word_id =  dict([(word,index) for index,(word,idf) in enumerate(list_dict)])
    over_all = list()
    for line in lines:
        term = line.split('<fff>')
        docs.append(term)
        words = term[2].split()
        words = [word for word in words if dict_idf.get(word,False)!= False ]
        each_line = []
        
       
        max_fre = max(words.count(word) for word in words)
        
        norm = 0
        for word in words:
           
            
            
            
            
            liqui = words.count(word)* 1./max_fre * float(dict_idf[word] )
            each_line.append((word_id[word],liqui))
            norm += liqui**2
        each_line = [ str(id)+':'+str(liqui/np.sqrt(norm)) for id,liqui in each_line]
        tf_idf = ' '.join(each_line)
        tf_idf = term[0]+'<ffff>'+term[1]+'<ffff>'+tf_idf
        over_all.append(tf_idf)
    string = '\n'.join(over_all)
    f = open('E:/tf_idf.txt','w')
    haha = '\n'.join([str(1), str(2)])
    
    f.write(string)
            
#tf_idf('E:/newtext.txt','E:/dict_idf.txt')



        

