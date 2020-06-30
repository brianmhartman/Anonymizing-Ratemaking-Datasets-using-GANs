# Save a module init file that contains a custom function that we'll use
# to verify that import works.

from __future__ import print_function

from torch.utils.data import Dataset, DataLoader

class PolicyDataset(Dataset):
    def __init__(self, filepath):
      #super.__init__(self)
      self.policy=pd.read_csv(filepath)
      features =pd.get_dummies(self.policy.loc[:,['Power', 'Brand', 'Gas', 'Region']])
      self.policy = pd.concat([self.policy.loc[:, ['ClaimNb', 'Exposure', 'CarAge', 'DriverAge', 'Density']], features], axis=1)


      #print(self.policy.head())
      #add error checking to make sure we got the file we want

    
    def __getitem__(self,index):
      #print(self.policy.head())
      return torch.from_numpy(self.policy.iloc[index].values).float()
      #Write a function that returns a tensor of the data
  
 
    def __len__(self):
      return len(self.policy.index)
    
    def getDatasetMeans(self):
      return(torch.from_numpy(self.policy.mean(axis=0).values)).float()
      
    def getDatasetSDs(self):
      return(torch.from_numpy(self.policy.std(axis=0).values)).float()