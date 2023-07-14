# importing libraries
import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt
import seaborn as sns                  # For plotting graphs
%matplotlib inline
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")

# loading the data
train = pd.read_csv("C:\\Users\\HP\Downloads\\Data_Science_Project_Problem_Statement (1)\\train.csv")
test = pd.read_csv("C:\\Users\\HP\\Downloads\\Data_Science_Project_Problem_Statement (1)\\test.csv")


train.columns


test.columns


train.shape, test.shape


# Print data types for each variable

train.dtypes

#printing first five rows of the dataset
train.head()



###univariate analysis

train['subscribed'].value_counts()


# plotting the bar plot of frequencies

train['subscribed'].value_counts().plot.bar()


sns.distplot(train["age"])

train['job'].value_counts().plot.bar()

train['default'].value_counts().plot.bar()


###bivariate analysis

print(pd.crosstab(train['job'],train['subscribed']))

job=pd.crosstab(train['job'],train['subscribed'])
job.div(job.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('Job')
plt.ylabel('Percentage')


print(pd.crosstab(train['default'],train['subscribed']))

default=pd.crosstab(train['default'],train['subscribed'])
default.div(default.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('default')
plt.ylabel('Percentage')


train['subscribed'].replace('no', 0,inplace=True)
train['subscribed'].replace('yes', 1,inplace=True)


corr = train.corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")


train.isnull().sum()


###model building

target = train['subscribed']
train = train.drop('subscribed',1)


# applying dummies on the train dataset

train = pd.get_dummies(train)


from sklearn.model_selection import train_test_split

# splitting into train and validation with 20% data in validation set and 80% data in train set.

X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state=12)


###logistic regression

from sklearn.linear_model import LogisticRegression

# defining the logistic regression model

lreg = LogisticRegression()


# fitting the model on  X_train and y_train

lreg.fit(X_train,y_train)


# making prediction on the validation set

prediction = lreg.predict(X_val)


from sklearn.metrics import accuracy_score



# calculating the accuracy score

accuracy_score(y_val, prediction)


###decesion tree

from sklearn.tree import DecisionTreeClassifier


# defining the decision tree model with depth of 4, you can tune it further to improve the accuracy score

clf = DecisionTreeClassifier(max_depth=4, random_state=0)


# fitting the decision tree model

clf.fit(X_train,y_train)


# making prediction on the validation set

predict = clf.predict(X_val)


# calculating the accuracy score

accuracy_score(y_val, predict)


test = pd.get_dummies(test)


test_prediction = clf.predict(test)


submission = pd.DataFrame()


# creating a Business_Sourced column and saving the predictions in it

submission['ID'] = test['ID']
submission['subscribed'] = test_prediction


submission['subscribed'].replace(0,'no',inplace=True)
submission['subscribed'].replace(1,'yes',inplace=True)


submission.to_csv("C:\\Users\\HP\\Downloads\\Data_Science_Project_Problem_Statement (1)\\test.csv", header=True, index=False)







