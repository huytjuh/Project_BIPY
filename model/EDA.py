import numpy as np 
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import seaborn as sns


class EDA:
  
    def __init__(self, data):
        self.data = data

    def pair_plot(self, bool_cat=True):
        
        data = self.data
        g = sns.pairplot(data=data, vars=data.columns)
        
        return g

