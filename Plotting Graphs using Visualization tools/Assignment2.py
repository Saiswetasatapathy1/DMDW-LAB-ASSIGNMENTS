#for import the files

import pandas as pd 
ds=pd.read_csv("Downloads/archive/iris.csv")
ds.head(5)


#mean for data 

import pandas as pd 
ds=pd.read_csv("Downloads/archive/iris.csv")
ds2=ds["sepal_width"].mean()
print("Mean is ",ds2)


#median for data
import pandas as pd 
ds=pd.read_csv("Downloads/archive/iris.csv")
ds2=ds["sepal_width"].median()
print("Median is ",ds2)


#maximum value of data
import pandas as pd 
ds=pd.read_csv("Downloads/archive/iris.csv")
ds2=ds["sepal_length"].max()
print("Maximum value is ",ds2)

#minimum value of data
import pandas as pd 
ds=pd.read_csv("Downloads/archive/iris.csv")
ds2=ds["sepal_width"].min()
print("Minimum value is ",ds2)

#mode for data
import pandas as pd 
ds=pd.read_csv("Downloads/archive/iris.csv")
ds2=ds["sepal_width"].mode()
print("Mode is ",ds2)


#box plot
import pandas as pd 
import matplotlib.pyplot as plot
ds=pd.read_csv("Downloads/archive/iris.csv")
b_plot = ds.boxplot(column = 'sepal_width')
b_plot.plot()
plot.show() 


#multiple box plot in single graph
import pandas as pd 
import matplotlib.pyplot as plot
ds=pd.read_csv("Downloads/archive/iris.csv")
b_plot = ds.boxplot(column = ['sepal_length','sepal_width','petal_width','petal_length'])
b_plot.plot()
plot.show()
