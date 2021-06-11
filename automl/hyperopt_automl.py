from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from hpsklearn import HyperoptEstimator
from hpsklearn import any_regressor
from hpsklearn import any_preprocessing
from hyperopt import tpe
import numpy as np
import pandas as pd
from collections import Counter
import re, os, sys, csv, math, operator
from sklearn import linear_model, svm, ensemble, tree, neighbors
from sklearn.metrics import mean_absolute_error, r2_score
import pickle, joblib

#Contains 86 elements (Without Noble elements as it does not forms compounds in normal condition)
elements = ['H','Li','Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe','Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
            'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
            'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er','Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 
            'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu' ]

elements_all = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 
                'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
                'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 
                'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
                'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np',
                'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg',
                'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn']



def RunMLModel(trainpath, testpath, dataproperty):
    train = pd.read_csv(trainpath) 
    test = pd.read_csv(testpath) 

    train_elem = train[elements]
    test_elem = test[elements] 

    new_x_train = train_elem.values
    new_x_test = test_elem.values
    
    new_x_train = np.asarray(new_x_train, dtype=np.float)
    new_x_test = np.asarray(new_x_test, dtype=np.float)
    
    new_x_train = np.asarray(new_x_train)
    new_x_test = np.asarray(new_x_test)

    y_train = train.pop(dataproperty).to_frame()
    y_test = test.pop(dataproperty).to_frame()

    new_y_train = np.array(y_train)
    new_y_test = np.array(y_test)

    new_y_train.shape = (len(new_y_train),)
    new_y_test.shape = (len(new_y_test),)

    filename = 'model_mlbrnet_{}.sav'.format(dataproperty)

    model = HyperoptEstimator(regressor=any_regressor('reg'), 
                              preprocessing=[], 
                              loss_fn=mean_absolute_error, 
                              algo=tpe.suggest, 
                              max_evals=1000, 
                              trial_timeout=600, 
                              verbose=False)

    # perform the search
    model.fit(new_x_train, new_y_train)

    # summarize performance
    y_pred = model.predict(new_x_test)
    mae = mean_absolute_error(new_y_test, y_pred)

    #Save model
    #pickle.dump(model, open(filename,'wb'))

    with open(printfilename, 'a') as f:
        sys.stdout = f
        print("MAE: %.4f" % mae)
        print(model.best_model())
        sys.stdout = original_stdout

original_stdout = sys.stdout
printfilename = 'result_hyperopt_automl_brnet.txt'

trainpath = '/path/to/training-set'
testpath = '/path/to/validation-set'
dataproperty = 'e_formation_energy'
with open(printfilename, 'a') as f:
    sys.stdout = f
    print('Material Property: {}'.format(dataproperty))
    sys.stdout = original_stdout
RunMLModel(trainpath, testpath, dataproperty)