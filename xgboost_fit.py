# -*- coding: utf-8 -*-
"""xgboost_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CIPqMCh0AxpNPuJNGpPAA-gJyA1CPZOI
"""

!pip install xgboost==1.5.1

from google.colab import drive
from typing import List
import numpy as np
import pandas as pd
import csv
import io
from typing import Tuple

def sequence_convertor(sequence_path: str) -> Tuple[np.array, np.array]:
  """
  Transform raw sequences into encoded numpy array
  :param sequence_path: ....
  :return: encoded numpy arrays
  """
  with open(sequence_path) as g4_csv:
    next(g4_csv)
    g4_reader = csv.reader(g4_csv, delimiter=',')
    sequences = []
    converted_sequences: List[List[int]] = []
    g4 = []
    for row in g4_reader:       
        if len(row[0]) < 30:
          mod_row = row[0]+'0'*(30-len(row[0]))
          sequences.append(mod_row)
          g4.append(row[1])
        elif len(row[0]) > 30:
          continue       
        else:
          sequences.append(row[0])
          g4.append(row[1])
    g4 = [int(flag) for flag in g4]

    for sequence in sequences:
      converted = []

      for base in sequence:
        if base == 'G':
            converted.append(-1)
        elif base == 'C':
            converted.append(1)
        else:
            converted.append(0)

      converted_sequences.append(converted)
    return np.array(converted_sequences), np.array(g4)

np_con_sequences, np_g4 = sequence_convertor('drive/MyDrive/dataset/modified_quadruplexes.csv')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

X = np_con_sequences
y = np_g4

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33)

xg_g4_decision_tree = XGBClassifier(max_depth=15, 
                                    learning_rate=0.06, 
                                    n_estimators=100, 
                                    gamma=0.01, 
                                    min_child_weight = 5,
                                    colsample_bytree = 0.9,
                                    num_parallel_tree=2)

xg_g4_decision_tree.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True)

xg_g4_decision_tree.save_model('drive/MyDrive/dataset/xgboost_kotrys.json')