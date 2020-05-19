#import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
	#importing Dataset
 	dataset = pd.read_csv('data.csv')
 	iv  = dataset.iloc[:,:-1].values #independent variables
 	dv = dataset.iloc[:,3].values #dependent variable
 	#takecare of missing data
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(iv[:, 1:3])
    iv[:, 1:3] = imputer.transform(iv[:, 1:3])
    
    #categorical data encoding
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    labelencoder_iv = LabelEncoder()
    iv[:,0]=labelencoder_iv.fit_transform(iv[:,0]) 
    onehotencoder = OneHotEncoder(categorical_features = [0])
    iv = onehotencoder.fit_transform(iv).toarray()
    labelencoder_dv  = LabelEncoder()
    dv = labelencoder_dv.fit_transform(dv);
	#splitting the dataset into training and testing datasets
    from sklearn.model_selection import train_test_split
    iv_train,iv_test,dv_train,dv_test = train_test_split(iv,dv,test_size=0.2,random_state=0)
    # Feature Scaling
	from sklearn.preprocessing import StandardScaler
	sc_X = StandardScaler()
	X_train = sc_X.fit_transform(X_train)
	X_test = sc_X.transform(X_test)
	sc_y = StandardScaler()
	y_train = sc_y.fit_transform(y_train)
    
    
    
    
    
    
    