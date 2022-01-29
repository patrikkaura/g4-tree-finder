# -*- coding: utf-8 -*-
"""
This python script uses XGBoost classifier to detect guanine quadruplexes in DNA sequences.
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
    _MERGING_INTERVAL_LENGTH: int = 15 

    def _sequence_convertor(self, *, sequence: str) -> np.array:
        """
        Convert sequences 
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
            if bool(prediction):
                results.append(index)
        return results

    def _create_intervals(self, predicted_position: List[int]) -> List[int]:
        """
        Create intervals used for merging
        :param predicted_position:
        :return:
        """
        intervals = [(i, i+self._MERGING_INTERVAL_LENGTH) for i in predicted_position]

        return intervals

    def _merge_results(self, *, results: List[Tuple[int]]) -> List[Tuple[int]]:
        """
        Return merged adjacent results from predict method
        :param results:
        :return:
        """
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
        Remove single occurance results from set
        :param merged_results:
        :return:
        """
        filtered = []
        for window in merged_results:
          if window[1] - window[0] > self._MERGING_INTERVAL_LENGTH:
            filtered.append(window)
        
        return filtered

    def _create_results(self, filtered_results, sequence) -> pd.DataFrame:
        """
        Format filtered results into Pandas dataframe
        :param filtered_results:
        :return:
        """
        df_data = {'Position': [position[0] for position in filtered_results],
                    'Sequence': [sequence[position[0]:position[1]+self._MERGING_INTERVAL_LENGTH] for position in filtered_results],
                    'Length': [position[1]+self._MERGING_INTERVAL_LENGTH-position[0] for position in filtered_results]}
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
        merg_intervals = self._create_intervals(predicted_position=predicted_position)
        
        merged_results = self._merge_results(results=merg_intervals)
        filtered_results = self._filter_results(merged_results=merged_results)

        return self._create_results(filtered_results, sequence)