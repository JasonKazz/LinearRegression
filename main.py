"""
Simple Linear Regression using Salary Dataset

Salary Dataset: https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression?resource=download
Simple Linear Regression finds the line of best fit for a given set of data with one independent variable.
The salary dataset uses the number of years of experience as the independent variable and the salary as the dependent variable.
A line of best fit is found by adjusting the weight m and bias b to minimize the error between the line of best fit and the data.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Getting data from the Salary Dataset

df = pd.read_csv('Salary_dataset.csv', index_col=[0]) # tells Pandas the first column is used to index or label the rows so it doesn't create an extra one
#print(df.index) # returns index column
#print(df.columns) # name of the columns
#print(df.values) # data

data = df.values # converts Pandas DataFrame to a NumPy array, stripping off index and column names, leaving just the data

X = data[:,0] # takes the independent variable column (features) (first column)
y = data[:,1] # takes the dependent variable column (target) (second column)

# Gradient Descent Algorithm

m = np.random.randint(1000) # random number between 0 and 999 (inclusive)
b = np.random.randint(1000)
m_list = [] # keeps track of m values as the algorithm repeats
b_list = []
e_list = []

n_epochs = 100 # number of epochs: # of times all of the data is run through the algorithm
lr = 0.01 # learning rate

for epoch in range(n_epochs):
  for i in range(len(X)):
    y_hat = m*X[i]+b # y_hat is the line of best fit
    de_dm = (y_hat-y[i])*X[i] # partial derivative of the error with respect to m
    de_db = (y_hat-y[i]) # partial derivative error with respect to b
    m = m-lr*de_dm # reevaluating m
    b = b-lr*de_db # reevaluating b
    e = .5*(m*X[i]+b - y[i])**2 # finding squared error
    m_list.append(m)
    b_list.append(b)
    e_list.append(e)
print(m_list)
print(b_list)

# Plotting the data and line of best fit

plt.scatter(X, y, color='red', label='given data') # X and y is the original data
plt.plot(X, m*X+b, label='line of best fit')
plt.grid()
plt.legend()
plt.xlabel('Experience in Years')
plt.ylabel('Salary')
plt.title('Finding the Line of Best Fit with SLR')
plt.savefig('Salary.pdf')
print("M: ",m)
print("B: ",b)
