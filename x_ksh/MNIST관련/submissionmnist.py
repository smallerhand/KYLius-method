import tensorflow as tf
import numpy as np

class img_pred:
    def __init__(self, opt_addr):
        # initialize/ load
        self.saver=tf.train.import_meta_graph(opt_addr+".meta")
        self.sess = tf.InteractiveSession()
        print("Meta_Graph Imported")
        
        # parameters save 
        self.saver.restore(self.sess, opt_addr)
        print("Parameters Restored")
        
        # variables 
        self.graph=tf.get_default_graph()
        self.X=self.graph.get_tensor_by_name('X:0')
        self.pred=self.graph.get_tensor_by_name('pred:0')
        self.p_keep_conv=self.graph.get_tensor_by_name('p_keep_conv:0')
        self.p_keep_hidden=self.graph.get_tensor_by_name('p_keep_hidden:0')
        print("Variables Saved")
    
    def number(self, dataarray):
        data = dataarray
        #classification result
        self.result=self.sess.run(self.pred, feed_dict={self.X: data, self.p_keep_conv: 1.0, self.p_keep_hidden: 1.0})
        return(self.result)

A=img_pred("/Users/kimseunghyuck/desktop/git/KYLius-method/x_ksh/optmnist/optmnist")

import pandas as pd
result=[]
arrayx=pd.read_csv("/Users/kimseunghyuck/desktop/git/KYLius-method/x_ksh/testmnist.csv")
arrayx[:5]
arrayx.shape #(28000, 784)
arrayx.iloc[1,:].reshape(-1, 784)

for i in range(28000):
    result.append(A.number(arrayx.iloc[i,:].reshape(-1, 784)))

path='/Users/kimseunghyuck/desktop/'

submission={}
i=0
for k in range(28000):
    submission[k+1]=result[i][0]
    i+=1

#파일 아웃풋
mnist1=pd.DataFrame([[k,v] for k,v in iter(submission.items())],columns=["ImageId","Label"])

mnist1.to_csv(path+'mnist1.csv', header=True, index=False, sep='\t')
