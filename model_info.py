import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model import Model


class ModelInfo:
    def __init__(self, model: Model, test_data: pd.DataFrame) -> None:
        self.test_data = test_data
        self.model = Model('model_info.csv')

        x = np.array([np.load(i) for i in test_data.image.values])

        self.test_data['model_pred_label'] = pd.Series(self.model(x,raw_output=False))
        self.model_pred_raw_x = pd.DataFrame(self.model(x,raw_output=True))


        self.confusion_matrix = confusion_matrix(self.test_data.label , self.test_data.model_pred_label)
        
        for i in range(10):
            self.test_data['model_pred_raw_{}'.format(i)] = self.model_pred_raw_x[i]
    def get_wrong_predictions(self):
        return self.test_data[self.test_data.label!=self.test_data.model_pred_label]

    def plot_conusion_matrix(self, figsize=(16, 10), dpi=300):
        fig, ax = plt.subplots(1,1,figsize = figsize , dpi = dpi)
        sns.heatmap(self.confusion_matrix, annot=True)
        return fig,ax
    def plot_process(self, idx: int, base_size=8):
        fig, ax = plt.subplots(1,10,figsize = (20,8) , dpi = 300)
        return

    def plot_brain(self):
        return