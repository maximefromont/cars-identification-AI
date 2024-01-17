from brand_model import basic_brand_model

#run the basic brand model for epoch from 10 to 100 with step 10
for epoch in range(10, 100, 10):
    basic_brand_model(epoch)

#read all the csv file with the pass brand-model/model-data/basic*

# Path: brand-model/brand-model-comparator.py
import pandas as pd
import glob

#read all the csv file with the pass brand-model/model-data/basic*
path = r'brand-model/model-data/basic*'
all_files = glob.glob(path)
li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

#concatenate all the csv file into one dataframe
frame = pd.concat(li, axis=0, ignore_index=True)

# Path: brand-model/brand-model-comparator.py
import matplotlib.pyplot as plt
import seaborn as sns

#plot the graph
sns.set(style="darkgrid")
sns.lineplot(x="epoch", y="accuracy", hue="model", data=frame)
plt.show()



    

