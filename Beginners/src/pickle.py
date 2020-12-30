#===Import dependencies===
import gc
import pickle
import psutil
import numpy as np
import pandas as pd
#===Function to track memory usage===
def memory_utilization():
  print('Current memory utilization: {}% ...'.format(psutil.virtual_memory().percent))
#===Create a dataframe using random integers between 0 and 1000===
memory_utilization()
var=pd.DataFrame(np.random.randint(0,1000,size=(int(2.5e8),2)),columns=['var1','var2'])
memory_utilization()
#==Create Pickle dump===
pickle.dump(var,open('var.pkl','wb'))
memory_utilization()
#===Delete the unused variable from memory===
del var
_=gc.collect()
memory_utilization()
#===Restore the variable from the disk for future use===
var=pickle.load(open('var.pkl','rb'))
memory_utilization()