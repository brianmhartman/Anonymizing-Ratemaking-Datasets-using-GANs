# Save a module init file that contains a custom function that we'll use
# to verify that import works.

from __future__ import print_function

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class PolicyDataset(Dataset):
    def __init__(self, data, cont_locs, small_test = None):
      self.policy = data.drop(data.columns[cont_locs], axis=1, inplace = False)  
      self.cont = data.iloc[:,cont_locs]
      self.small_test = small_test
      self.cont_locs = cont_locs

    def __getitem__(self,index):
      if len(self.cont_locs) > 0:
          return [torch.from_numpy(self.policy.iloc[index].values).float(),
                  torch.from_numpy(self.cont.iloc[index].values).float()]
      else:
          return [torch.from_numpy(self.policy.iloc[index].values).float(),
                  0]
    def __len__(self):
      if self.small_test is not None:
          return self.small_test
      return len(self.policy.index)
    
    def getDatasetMeans(self):
      return [(torch.from_numpy(self.policy.mean(axis=0).values)).float(),(torch.from_numpy(self.cont.mean(axis=0).values)).float()]
      
    def getDatasetSDs(self):
      return [(torch.from_numpy(self.policy.std(axis=0).values)).float(),(torch.from_numpy(self.cont.std(axis=0).values)).float()]
