# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15Z-IhHlxhQe6OTXpzTq9xKJPO7igPOVR
"""

from google.colab import drive
from typing import List
import numpy as np
import pandas as pd
import csv
from collections import defaultdict
import io

converted_sequences: List[List[int]] = []
g4 = []
def sequence_convertor(sequence_path: str) -> None:
  with open(sequence_path) as g4_csv:
    g4_reader = csv.reader(g4_csv, delimiter=',')
    sequences = [] 
    for row in g4_reader:
        sequences.append(row[1])
        g4.append(row[2])
    
    sequences.pop(0)
    g4.pop(0)

    for i in range(0, len(g4)):
        g4[i] = int(g4[i])

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

sequence_convertor('drive/MyDrive/dataset/generated_quadruplexes.csv')

np_con_sequences = np.array(converted_sequences)
np_g4 = np.array(g4)