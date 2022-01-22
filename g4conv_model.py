# -*- coding: utf-8 -*-
"""G4Conv_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19YoIegYTNS4nsT_QlN81bU3VXoHS6fN7
"""

!pip install xgboost==1.5.1

from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

import requests
import pandas as pd
import numpy as np

from time import sleep
from xgboost import XGBClassifier

class G4Conv():

    _FIXED_WINDOW_SIZE: int = 30 

    def _sequence_convertor(self, *, sequence: str) -> np.array:
        """
        Convert sequences with class defined _ENCODING 
        NOTE: don't change cause tree is trained to use exactly this parameters
        :param sequence: input sequence for conversion
        :return: numpy array with converted windows 
        """
        converted_sequences = []

        for i in range(0, len(sequence) - self._FIXED_WINDOW_SIZE):    
            converted = []
            for base in sequence[i:i+self._FIXED_WINDOW_SIZE]:
                if base == 'G':
                    converted.append(-1)
                elif base == 'C':
                    converted.append(1)
                else:
                    converted.append(0)
            converted_sequences.append(converted)
        return np.array(converted_sequences)

        # NECHAT
    def _init_tree(self, *, model_path: str) -> XGBClassifier:
        """
        Create model instance and load parameters from json model file
        :param model_path: path to file with model params in json
        :return: instance of gradient boosted tree
        """
        xgb = XGBClassifier()
        xgb.load_model(model_path)
        return xgb

    def _predict(self, *, model: XGBClassifier, converted_sequences: np.array) -> List[int]:
        """
        Return indexes with positive predictions
        :param model:
        :param converted_sequences:
        :return: 
        """
        results: List[int] = []
        predictions = model.predict(converted_sequences)
        predictions = list(predictions)

        for index, prediction in enumerate(predictions):
            if bool(prediction):   # pokud prediction == True
                results.append(index)
        return results

    def _merge_results(self, *, results: List[Tuple[int]]) -> List[Tuple[int]]:
        results = sorted(results, key=lambda x: x[0])
        i = 0
        for result in results:
          if result[0] > results[i][1]:
            i += 1
            results[i] = result
          else:
            results[i] = (results[i][0], result[1])
        return results[:i+1] 

    def _filter_results(self, *, merged_results: List[Tuple[int]]) -> List[Tuple[int]]:
        """
        doplnit
        """
        filtered = []
        for window in merged_results:
          if window[1] - window[0] > 15:
            filtered.append(window)
        
        return filtered

    def create_results(self, filtered_results, sequence):
        df_data = {'Position': [position[0] for position in filtered_results],
                    'Sequence': [sequence[position[0]:position[1]+15] for position in filtered_results],
                    'Length': [position[1]+15-position[0] for position in filtered_results]}
        return pd.DataFrame(df_data)

    def analyse(self, sequence: str, model_path: str) -> pd.DataFrame:
        """
        Analyse sequence for possible g4s
        :param sequence:
        :param model_path:
        :return:
        """
        model = self._init_tree(model_path=model_path)
        converted_sequences = self._sequence_convertor(sequence=sequence)
        predicted_position = self._predict(model=model, converted_sequences=converted_sequences)

        intervals = [(i, i+15) for i in predicted_position]
        merged_results = self._merge_results(results=intervals)
        filtered_results = self._filter_results(merged_results=merged_results)

        return self.create_results(filtered_results, sequence)

if __name__ == '__main__':
    sequence_file1 = open('drive/MyDrive/dataset/Human_papillomavirus.txt', 'r')
    sequence_file2 = open('drive/MyDrive/dataset/Neat1_chr11_65188245-65215011.txt', 'r')
    sequence_file3 = open('drive/MyDrive/dataset/Human_Mito.txt', 'r')

    results1 = G4Conv().analyse(
        sequence=sequence_file1.read(),
        model_path="drive/MyDrive/dataset/xgboost_kotrys.json",
    )

    results2 = G4Conv().analyse(
        sequence=sequence_file2.read(),
        model_path="drive/MyDrive/dataset/xgboost_kotrys.json",
    )

    results3 = G4Conv().analyse(
        sequence=sequence_file3.read(),
        model_path="drive/MyDrive/dataset/xgboost_kotrys.json",
    )

    results1.to_csv(path_or_buf='drive/MyDrive/dataset/papillomavirus.csv', index=False)
    results2.to_csv(path_or_buf='drive/MyDrive/dataset/neat1_chr11.csv', index=False)
    results3.to_csv(path_or_buf='drive/MyDrive/dataset/homo_sapiens_mitochondrion.csv', index=False)