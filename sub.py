import time
import os
import sys

import os
import datetime
import IPython
#import IPython.display
#import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tsfel
from sklearn.model_selection import train_test_split
from sklearn import neighbors,datasets
from sklearn import svm
from sklearn import tree
import seaborn as sn
import matplotlib.pyplot as plt
import pickle

import paho.mqtt.subscribe as subscribe
import paho.mqtt.client as mqtt
import matplotlib.pyplot as mpl
#from nn import *




#lecture des donnÃ©es pour IA
df=pd.read_csv("result.csv");
df1=pd.read_csv("result1.csv");
df2=pd.read_csv("result2.csv");

#mean
#df_mean = df.mean()
#df1_mean = df1.mean()
#df2_mean = df2.mean()


#std
#df_std = df.std()
#df1_std = df1.std()
#df2_std = df2.std()

#calcule
#df = (df - df_mean) / df_std
#df1 = (df1 - df1_mean) / df1_std
#df2 = (df2 - df2_mean) / df2_std


#Features extraction
fs=25000
cfg = tsfel.get_features_by_domain()
df = tsfel.time_series_features_extractor(cfg,df,window_size=50,window_spliter=True,overlap=0.5,fs=fs)
df1 = tsfel.time_series_features_extractor(cfg,df1,window_size=50,window_spliter=True,overlap=0.5,fs=fs)
df2 = tsfel.time_series_features_extractor(cfg,df2,window_size=50,window_spliter=True,overlap=0.5,fs=fs)

features =pd.concat([df, df1, df2])
feaures=features

#feaures.to_excel('out.xlsx', engine='xlsxwriter')  
#assigner les label
df=df.assign(label=1)
df1=df1.assign(label=2)
df2=df2.assign(label=3)


#mettre les data ensemble
DATA =pd.concat([df, df1, df2])


#separation fetures et labels et selection des features
labels=DATA['label']
features=features[['1_Max',
'1_Maximum frequency',
'1_Mean',
'1_Mean absolute deviation',
'1_Mean absolute diff',
'1_Median',
'1_Min',
'1_Root mean square',
'1_Skewness',
'1_Slope',
'1_Spectral centroid',
'1_Spectral decrease',
'1_Spectral distance',
'1_Spectral kurtosis',
'1_Spectral skewness',
'1_Spectral slope',
'1_Spectral spread',
'1_Spectral variation',
'1_Total energy',
'1_Variance',
'1_Wavelet absolute mean_0',
'1_Wavelet absolute mean_1',
'1_Wavelet absolute mean_2',
'1_Wavelet absolute mean_3',
'1_Wavelet absolute mean_4',
'1_Wavelet absolute mean_5',
'1_Wavelet absolute mean_6',
'1_Wavelet absolute mean_7',
'1_Wavelet absolute mean_8',
'1_Wavelet energy_0',
'1_Wavelet energy_1',
'1_Wavelet energy_2',
'1_Wavelet energy_3',
'1_Wavelet energy_4',
'1_Wavelet energy_5',
'1_Wavelet energy_6',
'1_Wavelet energy_7',
'1_Wavelet energy_8',
'1_Wavelet entropy',
'1_Wavelet standard deviation_0',
'1_Wavelet standard deviation_1',
'1_Wavelet standard deviation_2',
'1_Wavelet standard deviation_3',
'1_Wavelet standard deviation_4',
'1_Wavelet standard deviation_5',
'1_Wavelet standard deviation_6',
'1_Wavelet standard deviation_7',
'1_Wavelet standard deviation_8',
'1_Wavelet variance_0',
'1_Wavelet variance_1',
'1_Wavelet variance_2',
'1_Wavelet variance_3',
'1_Wavelet variance_4',
'1_Wavelet variance_5',
'1_Wavelet variance_6',
'1_Wavelet variance_7',
'1_Wavelet variance_8',
'1_Wavelet entropy',
'1_Zero crossing rate']]

print(features.shape)


#label.to_excel('label.xlsx', engine='xlsxwriter')  
#features.to_excel('featuress.xlsx', engine='xlsxwriter')  



#split into train and test data
X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=1, train_size = 0.75)

#classifier
model=tree.DecisionTreeClassifier();
model1=svm.SVC(kernel='linear',C=1.0);
model2=neighbors.KNeighborsClassifier();

#training
model.fit(X_train,y_train)
model1.fit(X_train,y_train)
model2.fit(X_train,y_train)




#prediction
predit=model.predict(X_test);
predit1=model1.predict(X_test);
predit2=model2.predict(X_test);

#evaluation de model par la creation de matrice de confusion
confusion_matrix = pd.crosstab(y_test ,predit , rownames=['Actual'], colnames=['Predicted'])
confusion_matrix1 = pd.crosstab(y_test ,predit1 , rownames=['Actual'], colnames=['Predicted'])
confusion_matrix2 = pd.crosstab(y_test ,predit2 , rownames=['Actual'], colnames=['Predicted'])

#affichage des matrices de confusion
sn.heatmap(confusion_matrix, annot=True)
plt.title(f"Accuracy: {sum(predit == y_test) / y_test.shape[0]}")
plt.figure()
plt.title(f"Accuracy: {sum(predit1 == y_test) / y_test.shape[0]}")
sn.heatmap(confusion_matrix1, annot=True)
plt.figure()
plt.title(f"Accuracy: {sum(predit2 == y_test) / y_test.shape[0]}")
sn.heatmap(confusion_matrix2, annot=True)
plt.show()







#connexion avec mosquitto
ip_broker = "localhost"   #adresse de reception des données
port_broker = 1883        #port mosquitto
data = ""
temps,mesure,correctlabel = [],[],[]
xlist=[]


    
def on_message_print(client, userdata, message):
    #print(message.topic + ": " + str(message.payload.decode("utf-8")))
    valeur = message.payload.decode("utf-8")
    valeur=valeur.split(";")
    xlist.append(valeur) 
    temps.append(float(valeur[0]))
    mesure.append(float(valeur[1]))
    correctlabel.append(float(valeur[2]))
    #data=pd.DataFrame(list(zip(temps, mesure)), columns = ['time', 'mesure'])
    #print(data)
    tempss = pd.DataFrame(temps)
    train_df = pd.DataFrame(mesure, columns = ['mesure'])
    correctlabelS=pd.DataFrame(correctlabel)
    
    class WindowGenerator():
      def __init__(self, input_width, label_width, shift,
                   train_df=train_df,
                   label_columns=None):
        # Store the raw data.
        self.train_df = train_df
       
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
        def __repr__(self):
            return '\n'.join([
                f'Total window size: {self.total_window_size}',
                f'Input indices: {self.input_indices}',
                f'Label indices: {self.label_indices}',
                f'Label column name(s): {self.label_columns}'])


    #generation de deux fenetres glissant de deux taille differents
    w1 = WindowGenerator(input_width=50, label_width=1, shift=1,
                         label_columns=['Mesure'])
    


    def split_window(self, features):
      inputs = features[:, self.input_slice, :]
      # Slicing doesn't preserve static shape information, so set the shapes
      # manually. This way the `tf.data.Datasets` are easier to inspect.
      inputs.set_shape([1, self.input_width, 1])
      return inputs

    WindowGenerator.split_window = split_window


    def plotmauvais(self, model=None, plot_col='mesure', max_subplots=3):
      inputs = self.example
      plt.figure(figsize=(12, 8))
      plot_col_index = self.column_indices[plot_col]
      max_n = min(max_subplots, len(inputs))
      
      for n in range(max_n):
        plt.subplot(2, 1, n+1)
        plt.ylabel('mesure')
        
        
        plt.plot(train_df)
        plt.axvline(i,color= 'red')
        plt.axvline(i+self.input_width,color= 'red')
       
        #plt.plot(self.input_indices+i, inputs[n, :, plot_col_index],'r',
                 #label=label, marker='.', zorder=-10)
        
        #plt.axvline(self.input_indices[],color= 'red')
        plt.subplot(2, 1, n+2)
        plt.ylabel('mesure')
        plt.plot(self.input_indices+i, inputs[n, :, plot_col_index],
                 label='label 3', marker='.', zorder=-10)
     
        
       

      if n == 0:
          plt.legend()

      plt.xlabel('Time')
    WindowGenerator.plotmauvais = plotmauvais


    def plotbon(self, model=None, plot_col='mesure', max_subplots=3):
      inputs = self.example
      plt.figure(figsize=(12, 8))
      plot_col_index = self.column_indices[plot_col]
      max_n = min(max_subplots, len(inputs))
      
      for n in range(max_n):
        plt.subplot(2, 1, n+1)
        plt.ylabel('mesure')
        
        
        plt.plot(train_df)
        plt.axvline(i,color= 'red')
        plt.axvline(i+self.input_width,color= 'red')
       
        #plt.plot(self.input_indices+i, inputs[n, :, plot_col_index],'r',
                 #label=label, marker='.', zorder=-10)
        
        #plt.axvline(self.input_indices[],color= 'red')
        plt.subplot(2, 1, n+2)
        plt.ylabel('mesure')
        plt.plot(self.input_indices+i, inputs[n, :, plot_col_index],
                 label='label 1', marker='.', zorder=-10)
     
        
       

      if n == 0:
          plt.legend()

      plt.xlabel('Time')
    WindowGenerator.plotbon = plotbon
    
    
    
    
    def plotmoy(self, model=None, plot_col='mesure', max_subplots=3):
      inputs = self.example
      plt.figure(figsize=(12, 8))
      plot_col_index = self.column_indices[plot_col]
      max_n = min(max_subplots, len(inputs))
      
      for n in range(max_n):
        plt.subplot(2, 1, n+1)
        plt.ylabel('mesure')
        
        
        plt.plot(train_df)
        plt.axvline(i,color= 'red')
        plt.axvline(i+self.input_width,color= 'red')
       
        #plt.plot(self.input_indices+i, inputs[n, :, plot_col_index],'r',
                 #label=label, marker='.', zorder=-10)
        
        #plt.axvline(self.input_indices[],color= 'red')
        plt.subplot(2, 1, n+2)
        plt.ylabel('mesure')
        plt.plot(self.input_indices+i, inputs[n, :, plot_col_index],
                 label='label 2', marker='.', zorder=-10)
     
        
       

      if n == 0:
          plt.legend()

      plt.xlabel('Time')
    WindowGenerator.plotmoy = plotmoy
    
    
    
    ITER=len(train_df)
    # Stack three slices, the length of the total window.
    for i in range(ITER):
        if(i%1==0):
           
            example_window = tf.stack([np.array(train_df[i:i+w1.total_window_size])])
            #print(example_window.shape)
            if(example_window.shape[1]==50):
                example_inputs= w1.split_window(example_window)
                #print('oui ')
                w1.example = example_inputs
                print(train_df.shape)
                cfg = tsfel.get_features_by_domain()
                computeFeature = tsfel.time_series_features_extractor(cfg, w1.example,window_spliter=True,fs=fs)#extraction des features
                #computeFeature.to_excel('transS.xlsx', engine='xlsxwriter')  
                computeFeature=computeFeature[['0_Max',
                '0_Maximum frequency',
                '0_Mean',
                '0_Mean absolute deviation',
                '0_Mean absolute diff',
                '0_Median',
                '0_Min',
                '0_Root mean square',
                '0_Skewness',
                '0_Slope',
                '0_Spectral centroid',
                '0_Spectral decrease',
                '0_Spectral distance',
                '0_Spectral kurtosis',
                '0_Spectral skewness',
                '0_Spectral slope',
                '0_Spectral spread',
                '0_Spectral variation',
                '0_Total energy',
                '0_Variance',
                '0_Wavelet absolute mean_0',
                '0_Wavelet absolute mean_1',
                '0_Wavelet absolute mean_2',
                '0_Wavelet absolute mean_3',
                '0_Wavelet absolute mean_4',
                '0_Wavelet absolute mean_5',
                '0_Wavelet absolute mean_6',
                '0_Wavelet absolute mean_7',
                '0_Wavelet absolute mean_8',
                '0_Wavelet energy_0',
                '0_Wavelet energy_1',
                '0_Wavelet energy_2',
                '0_Wavelet energy_3',
                '0_Wavelet energy_4',
                '0_Wavelet energy_5',
                '0_Wavelet energy_6',
                '0_Wavelet energy_7',
                '0_Wavelet energy_8',
                '0_Wavelet entropy',
                '0_Wavelet standard deviation_0',
                '0_Wavelet standard deviation_1',
                '0_Wavelet standard deviation_2',
                '0_Wavelet standard deviation_3',
                '0_Wavelet standard deviation_4',
                '0_Wavelet standard deviation_5',
                '0_Wavelet standard deviation_6',
                '0_Wavelet standard deviation_7',
                '0_Wavelet standard deviation_8',
                '0_Wavelet variance_0',
                '0_Wavelet variance_1',
                '0_Wavelet variance_2',
                '0_Wavelet variance_3',
                '0_Wavelet variance_4',
                '0_Wavelet variance_5',
                '0_Wavelet variance_6',
                '0_Wavelet variance_7',
                '0_Wavelet variance_8',
                '0_Wavelet entropy',
                '0_Zero crossing rate']]
                                                     
                
                
                z=model.predict(computeFeature);
              
             
               
                if(z==1):
                    w1.plotbon()
                    #plt.legend(str(correctlabel[correctlabelS.shape[1]] ))
                    #print(correctlabel[correctlabelS.shape[1]])
                    
                elif(z==2):
                    w1.plotmoy()
                    #plt.legend(str(correctlabel[correctlabelS.shape[1]]))
                    #print(str(correctlabel[correctlabelS.shape[1]]))
                elif(z==3):
                    w1.plotmauvais()
                    #plt.legend(str(correctlabel[correctlabelS.shape[1]]))
                    #print(str(correctlabel[correctlabelS.shape[1]]))
                plt.show()
                #print(f"vari label {correctlabel[1]} label predit {z}")
                print(correctlabel)
                
       
      
        
capteur = subscribe.callback(on_message_print, "#", hostname=ip_broker, port=port_broker)     
    
   
    


