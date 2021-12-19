# -*- coding: utf-8 -*-
"""konverze.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15Z-IhHlxhQe6OTXpzTq9xKJPO7igPOVR
"""

from google.colab import drive
from typing import List
import numpy as np
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
        sequences.append(row[1])
        g4.append(row[2])

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

[np_con_sequences, np_g4] = sequence_convertor('drive/MyDrive/dataset/generated_quadruplexes.csv')