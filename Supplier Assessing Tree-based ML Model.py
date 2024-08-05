# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:03:55 2024

@author: Chiawei.Kuo07
"""

### Part 0: Set up

## Part 0.1: Library Importing

#utility functions 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import tqdm
import pickle
import random 

warnings.filterwarnings("ignore")

#data preprocessing 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
#metrics
from sklearn.metrics import make_scorer


#models 
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

#cross-validation
from sklearn.model_selection import cross_val_score

#hyper parameter tuning
import time
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#Setting the random seed 
seed = 42
np.random.seed(seed) 
random.seed(seed)


## Part 0.2: Data Importing

cost = pd.read_csv("cost.csv")
cost.head()

cost.info()

print(len(cost["Task ID"].unique()), len(cost["Supplier ID"].unique()))

cost.isna().sum()

suppliers = pd.read_csv("suppliers.csv") 
suppliers.head()


suppliers.info()

suppliers.isna().any().sum()


tasks= pd.read_excel("tasks.xlsx") 
tasks.head()


tasks.columns

tasks.info()

tasks.isna().any().sum()

#checking merging conditions on task id 
cost["Task ID"]


#checking merging conditions on task id 
tasks.index

cost["Task ID"] = [x[-4:] + " " + x[3:5] + " " + x[0:2] for x in cost["Task ID"].to_list()]
cost["Task ID"]



### Part 1: Data Preparation
## Part 1.1: Finding Missing Values


#1.1-finding missing values
print("1. There are",cost.isna().any().sum(),"missing values in cost")
print("2. There are",suppliers.isna().any().sum(),"missing values in suppliers")
print("3. There are",tasks.isna().any().sum(),"missing values in tasks")

#1.1-count the number of tasks, suppliers, features and cost values in all data sets
print("4. There are",tasks.shape[0],"tasks")
print("5. There are",suppliers.shape[0],"suppliers")
print("6. There are",suppliers.shape[1]-1,"features in suppliers")
print("7. There are",tasks.shape[1]-1,"features in tasks")
print("8. There are",len(cost['Cost'].unique()),"costs")


print(len(cost['Cost'])-len(cost['Cost'].unique()))


print(pd.Series([len(x) for x in tasks['Task ID']]).value_counts())


tasks.loc[pd.Series([len(x) for x in tasks['Task ID']]) == 11]



#idenitying the unique tasks id in cost and tasks
tasks=tasks.replace("2021 11 05 ","2021 11 05") #bc there is a space in task file, but no space in cost file
tasks.loc[tasks["Task ID"] == "2021 11 05"]
cost_task_id = cost["Task ID"].unique()
tasks_task_id = tasks["Task ID"].unique()
print(f"there are {len(cost_task_id)} unique Task ID in cost")
print(f"there are {len(tasks_task_id)} unique Task ID in task")


#identifying and removing the ids that need to be removed
to_drop_list = []

for cur_id in tasks_task_id:
    if cur_id not in cost_task_id:
        to_drop_list.append(cur_id)
    else:
        continue 

print("These",len(to_drop_list),"Task IDs need to be removed")
print(to_drop_list)
#originally removed 11 tasks but now 10 tasks-->
#refer to the previous cell regarding the spacing issue of Task ID "2021 11 05 " in task file


#removing these IDs from the the tasks dataframe
#set the index 
tasks.index = tasks["Task ID"]
#removing the lsit through the matching index
tasks = tasks.drop(to_drop_list)
#reindex
tasks = tasks.reset_index(drop=True)
print("\ntasks after deleting 10 Task IDs:\n")


tasks.info()


## Part 1.2: Calculating Five Number Summary

#retrieving five number summary for variables from supplier
suppliers_sum = suppliers.drop(columns = ["Supplier ID"]).agg(["max","min","mean","var"]).transpose()
suppliers_sum.head()


#retrieving the five number summary for variables from tasks
tasks_sum = tasks.drop(columns = ["Task ID"]).agg(["max","min","mean","var"]).transpose()
tasks_sum.head()


#merging both dataframes together 
summary_df = pd.concat([suppliers_sum, tasks_sum])
summary_df.insert(0, "factor", summary_df.index)
summary_df = summary_df.reset_index(drop = True)
summary_df



#identifying variables that need to be removed
to_drop_var_task = summary_df[summary_df["var"] < 0.01]["factor"].to_list()
print("The following",len(to_drop_var_task),"TFs(task features) need to be removed as it's variance is <0.01")
print(to_drop_var_task)



#removing the variables from tasks following above 
#tasks=tasks.drop(['TF31', 'TF75', 'TF79', 'TF84', 'TF88', 'TF92', 'TF96', 'TF100', 'TF104', 'TF108', 'TF112'],axis=1)
tasks=tasks.drop(to_drop_var_task,axis=1)
#tasks=tasks[~tasks["Task ID"].isin(to_drop_var_task)]
print("\ntasks after deleting the TF(task feature) with variance <0.01:\n")
print(tasks)


## Part 1.3: Scale all features to [-1,1]

# Tasks

#select features without Task ID in the dataframe to allow to be scaled 
new_tasks=tasks.iloc[:,1:].values
new_tasks


#values after min max scaling
scaler=preprocessing.MinMaxScaler(feature_range=(-1,1))
scaled_tasks=scaler.fit_transform(new_tasks)
scaled_tasks

#save the scaled data to tasks
tasks_temp=pd.DataFrame(scaled_tasks)
tasks_temp.columns=tasks.columns[1:]
tasks_temp.insert(0,"Task ID" ,tasks["Task ID"])
tasks=tasks_temp
tasks.describe()

# Suppliers

#select features without Supplier ID in the dataframe to allow to be scaled 
new_suppliers=suppliers.iloc[:,1:].values
new_suppliers


#values after min max scaling
scaler=preprocessing.MinMaxScaler(feature_range=(-1,1))
scaled_suppliers=scaler.fit_transform(new_suppliers)
scaled_suppliers=pd.DataFrame(scaled_suppliers)

#save the scaled data to suppliers
suppliers_temp=pd.DataFrame(scaled_suppliers)
suppliers_temp.columns=suppliers.columns[1:]
suppliers_temp.insert(0,"Supplier ID" ,suppliers["Supplier ID"])
suppliers=suppliers_temp
suppliers.describe()


## Part 1.4: Computing Absolute Correlations

#1.4-caclucating absolute correlation and visualising it
corr_df = tasks.corr().abs()
import seaborn as sn
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [50, 20]
plt.rcParams["figure.autolayout"] = True
sn.heatmap(corr_df,annot=True, cmap = 'Blues')
plt.title('absolute correlation of all pairs of features', fontsize=40)
plt.show()


#initialisation 
task_var = list(tasks.columns[1:].copy())  
#copy tasks' all columns(first row) and convert them into a list-->a list that has all the TFs
threshold = 0.8 
dropped_var = [] #create a list that contain the TFs that needed to be dropped

#retrieving the current dataframe
cur_df = tasks[task_var] 
#put tasks' TFs(from task_var list, which has all the TFs) list into "cur_df" 

#unstacking the correlation matrix 
corr_df = tasks[task_var].corr().abs() 
#make tasks.csv's TFs into a absolute correlation and renamed to "corr_df"

#retrieving correlation pairs 
corr_pairs = tasks[task_var].corr().unstack().reset_index() 
#.unstack() makes the correlation dataframe into another form (var_1 var_2a correlation, var_1 var_2b correlation....) 
#.reset_index() makes the new form into a dataframe that puts var_1 into a column named level_0, 
#and var_2 into a column named level_1 and correlation into a column named 0
corr_pairs.columns = ["variable_1", "variable_2", "correlation"] 
#rename the columns of previous line of code, so the column names are changed from 
#level_0, level_1, 0 into variable_1, variable_2, correlation
corr_pairs["abs_correlation"] = abs(corr_pairs["correlation"]) 
#create a new column called abs_correlation by attach a ['abs_correlation'] 
#the rows of value under this column are the dataframe's "correlation" column's absolute correlations (by using abs())

#droppping the perfectly correlated pairs 
corr_pairs = corr_pairs.loc[~(corr_pairs["variable_1"] == corr_pairs["variable_2"]),:] 
#drop the rows that variables are compared with themselves, e.g. TF1 and TF1 (this kind of rows always have correlation of "1")

#sorting 
corr_pairs = corr_pairs.sort_values(by = ["abs_correlation"], ascending = False).reset_index(drop = True) 
#sort the dataframe in an descending order so that we get the TFs that have highest absolute correlations at top

#retrieving the pairs > threshold 
corr_pairs_filtered = corr_pairs[corr_pairs["abs_correlation"]>=threshold] 
#create a new dataframe called "corr_pairs_filtered" and put TFs that have abs correlation>0.8 into this new dataframe

while len(corr_pairs_filtered) != 0: #execute this while-loop until len(corr_pairs_filtered)=0
    var_1 = list(corr_pairs_filtered["variable_1"])[0] 
    #the highest TF of variable_1 in "corr_pairs_filtered" which shows on the top of the dataframe in row 0
    var_2 = list(corr_pairs_filtered["variable_2"])[0] 
    #the highest TF of variable_2 in "corr_pairs_filtered" which shows on the top of the dataframe in row 0
    
    exclusion_list = [var_1, var_2] #put the highest TFs of var_1 and var_2 into "exclustion_list"
    exclusion_list.extend(dropped_var) #put values in "dropped_var" into "exclustion_list"
    
    corr_1 = corr_df[var_1].sort_values(ascending = False) #sort the corr_df's variable_1 in descending order
    corr_1 = corr_1[~corr_1.index.isin(exclusion_list)] #check whether these TFs are in "exclusiton_list"
    
    corr_2 = corr_df[var_2].sort_values(ascending = False) 
    corr_2 = corr_2[~corr_2.index.isin(exclusion_list)]
    corr_1[3]
    corr_2[3] 
    
    for val_1, val_2 in zip(corr_1, corr_2): 
        #zip(corr_1,crr_2) put the values in the same index of corr_1 and corr_2 together as a pair

        #print("{0:.16f}".format(float(val_1)))
        #print("{0:.16f}".format(float(val_2)))
        val_1=("{0:.16f}".format(float(val_1)))
        val_2=("{0:.16f}".format(float(val_2)))
        if val_1> val_2: #in the pairs, if val_1 > val_2, 
            drop_var = var_1 #put var_1 into "drop_var" -->drop the most correlated ones
            print(f"remove: {drop_var} {val_1} \n keep: {var_2} {val_2}")
            corr_df = corr_df.drop(drop_var,axis=1)
            corr_df = corr_df.drop(drop_var,axis=0)
            break

        elif val_1< val_2: #in the pairs, if val_1 < val_2,
            drop_var = var_2 #put var_2 into "drop_var" -->drop the most correlated ones
            print(f"remove: {drop_var} {val_2} \n keep: {var_1} {val_1}")
            corr_df = corr_df.drop(drop_var,axis=1)
            corr_df = corr_df.drop(drop_var,axis=0)
            break
        else:
            continue  #if val_1, val_2 not matching the above, skip them and continue to look the next val_1 and val_2 

    #adding the dropped variable 
    dropped_var.append(drop_var) #put all the TFs "drop_var" that are most correlated into "dropped_var"
    
    #updating the corr_pairs_filtered by removing the variable 1 and variable 2 both present case 
    corr_pairs_filtered = corr_pairs_filtered[~(corr_pairs_filtered.variable_1.isin([var_1, var_2]) & 
                                                corr_pairs_filtered.variable_2.isin([var_1, var_2]))] 

    #updating the corr_pairs_filtered by removing the dropped variable present case 
    corr_pairs_filtered = corr_pairs_filtered[~(corr_pairs_filtered.variable_1.isin([drop_var]) | corr_pairs_filtered.variable_2.isin([drop_var]))] 
    #update the corr_pairs_filtered which exclude those TFs where the TFs of variable_1 in "corr_pairs_filtered is in [drop_var] 
    #OR the TFs of variable_2 in "corr_pairs_filtered is in [drop_var]
#print out the final absolute correlation
corr_df



plt.rcParams['figure.figsize'] = [50,20]
plt.rcParams['figure.autolayout'] =True
plt.title(' absolute correlation of all pairs of features after dropped', fontsize=40)
sn.heatmap(corr_df,annot=True, cmap = 'Blues')
plt.show()


## Part 1.5: Top 20 Suppliers for each task

#retrieving the unique suppliers list 
supplier_list = cost["Supplier ID"].unique()

#creating an empty supplier list 
supplier_top_20_list = []

#creating a group by object 
groupby_ID = cost.groupby("Task ID")

#for loop to loop through each of the task ID's
for name, group in groupby_ID:
    #sort by ascending cost and retrieve the lowest 20 supplier ID's
    cur_top_20 = list(group.sort_values(by = ['Cost'], ascending = True)["Supplier ID"][:20].values)
    
    #save these to the supplier list 
    supplier_top_20_list.extend(cur_top_20)

#creating a frequency table of number of times in top 20 
print("The top 10 that has appeared the most in the top 20 lists")
print(pd.Series(supplier_top_20_list).value_counts().head(10))

print("\nThe lowest 10 that has appeared the least in the top 20 lists")
print(pd.Series(supplier_top_20_list).value_counts().tail(10))
    
#retrieving the unique set for the supllier that has been in the top 20 
unique_supplier = list(set(supplier_top_20_list))
print(f"\nthere has been {len(unique_supplier)} suppliers that has been a part of the top 20")

#retrieving the non-overlapping elements 
print(f"\nthe following suppliers have not been a part in the top 20 list")
to_drop_suppliers = list(set(supplier_list) -set(unique_supplier))
print(to_drop_suppliers)

#remove from the data that all suppliers that never appear in the top 20 of any task
cost.index = cost["Supplier ID"]
print(f"there are {len(cost['Supplier ID'].unique())} suppliers before drop in cost.csv")
cost = cost.drop(to_drop_suppliers)
cost = cost.reset_index(drop=True)
print(f"there are {len(cost['Supplier ID'].unique())} rows before drop in cost.csv")

suppliers.index = suppliers['Supplier ID']
print(f"there are {len(suppliers['Supplier ID'].unique())} suppliers before drop in suppliers.csv")
suppliers = suppliers.drop(to_drop_suppliers) 
suppliers = suppliers.reset_index(drop=True)
print(f"there are {len(suppliers['Supplier ID'].unique())} suppliers after drop in suppliers.csv")



## Part 1.6: Finalizing the suppliers and tasks dataframe


#save the prepared supplier dataframe under this name 
cleaned_suppliers = suppliers
cleaned_suppliers.head()


cleaned_suppliers.info()


tasks.index = tasks['Task ID']
cleaned_tasks = tasks.copy()
cleaned_tasks = tasks_temp[corr_df.columns.values]
cleaned_tasks = cleaned_tasks.reset_index()


#save the prepared tasks dataframe under this name 
cleaned_tasks.head()


cleaned_tasks.info()


#save the prepared cost dataframe under this name 
cleaned_cost = cost
cleaned_cost.head()


cleaned_cost.info()



### Part 2: Exploratory Data Anlaysis (EDA)

## Part 2.1: Distribution of feature values for each task

#create the copy for tasks dataset
eda_tasks = cleaned_tasks.copy()
#change the index of the copy dataset
eda_tasks.index = eda_tasks['Task ID']
#drop specified labels from columns
eda_tasks.drop('Task ID',axis=1,inplace=True)
#transpose the copy dataset
eda_tasks = eda_tasks.transpose()
eda_tasks


#decide the figure size and autolayout
plt.rcParams["figure.figsize"] = [50, 20]
plt.rcParams["figure.autolayout"] = True
#Create a boxplot that shows the distribution of feature values for each task by for loop
for col in eda_tasks.columns:
    eda_tasks[col] = pd.to_numeric(eda_tasks[col])
eda_tasks.boxplot()
#add title for boxplot
plt.title('The distribution of feature values for each task', fontsize=40 )
#rotate the x-axis lable 
plt.xticks(rotation=-45)
plt.show()


## Part 2.2: Distribution of Errors in Naive Model

#copy cleaned_cost to eda_cost
eda_cost = cleaned_cost.copy()

#create dataframe for minimum cost of each task
min_cost = pd.DataFrame(eda_cost.groupby(['Task ID'])['Cost'].min())

#set Task ID as index
eda_cost.index = eda_cost['Task ID']
eda_cost.drop(['Task ID'],axis=1, inplace = True)

#check whether the task id in both dataframe match
print(eda_cost.index.unique() == min_cost.index)

#calculate the error by automatic index alignment
eda_cost['Error'] = min_cost['Cost'] - eda_cost['Cost']



#calculate RMSE for each supplier
eda_cost['Error_square'] = eda_cost['Error']**2
sum_of_square = eda_cost.groupby('Supplier ID')['Error_square'].agg('sum')
num = eda_cost.groupby('Supplier ID')['Error_square'].agg('count')
rmse = (sum_of_square/num)**0.5

#sort RMSE in ascending order
rmse = rmse.reset_index()
rmse = rmse.rename(columns = {'Error_square':'RMSE'})
rmse = rmse.sort_values(by='RMSE')
rmse = rmse.reset_index()
rmse.drop(['index'],axis=1,inplace=True)

#get Supplier ID order in ascending order of RMSE
order = rmse['Supplier ID']


#get error distribution of suppliers
group = eda_cost.groupby('Supplier ID')['Error'].apply(list)

#sort the suppliers in ascending order of RMSE
group = group.reindex(index = order)



#set plot size
fig, ax = plt.subplots(figsize=(50, 20))

#set title & labels
ax.set_title('Native Error Distributions', size=15)
ax.set_xlabel('Supplier',size=15)
ax.set_ylabel('Error',size=15)

#set x-axis labels
labels = group.index.to_list()
ax.set_xticklabels(labels, size=15)

#set gridlines
ax.grid(True, 'major', 'both', ls='--', lw=.5, c='k', alpha=.3)

#annotate the RMSE for each supplier
height = -0.24
width = np.arange(0.7,64.8,1)

for i in range(len(rmse)):
    plt.annotate(round(rmse['RMSE'][i], 3), xy=(width[i], height), size=14)

#plot box plot
plt.boxplot(group)

#show plot
plt.show()


print(f"The mean of RMSE: {rmse['RMSE'].mean()}" )
print(f"The median of RMSE: {rmse['RMSE'].median()}")
print(f"The lowest RMSE: {rmse['RMSE'].min()}")
print(f"The highest RMSE: {rmse['RMSE'].max()}")


## Part 2.3: Cost values matrix

#change the dataset to wide
c_hmp = cleaned_cost.pivot(index='Task ID', values = 'Cost', columns = 'Supplier ID')
#create a heatmap shows the cost values as a matrix of tasks (rows) and suppliers (columns)
sn.heatmap(c_hmp,annot=True,cmap='Blues')
#add the title for heatmap
plt.title('The cost values as a matrix of tasks and suppliers', fontsize=40)
plt.show()


## Part 2.4: Extra Plots

# Added plot (Line Graph for supplier and cost)

#change the dataset to wide
tstpp = cleaned_cost.pivot(index='Task ID',columns='Supplier ID',values='Cost')
#create a copy for wide dataset
tstp = tstpp.copy()
#create a copy with droping 'S3' from columns
tstp3 = tstp.drop("S3", axis='columns')
#create a copy with removing 'S3'
tstps3 = tstp.pop('S3')

#decide the figure size and autolayout
plt.rcParams['figure.figsize'] = [50,20]
plt.rcParams["figure.autolayout"] = True
#decide the ponit shape 
markerss = ['d', 'v', 's', '', '^', 'o', 'X', 'D', '', '<','p']
#create a Line Graph for supplier and cost
axl = tstp3.plot(kind='line')
#high light the 'S3' on graph
axxl = tstps3.plot(kind='line',linewidth=6, color='blue') 


#assign the shape by for loop
for i, line in enumerate(axl.get_lines()): 
    line.set_marker(markerss[i % len(markerss)]) 
    axl.legend()
 
    

#add lines on graph
plt.grid()
#add title for graph
plt.title('Suppliers’ Costs for Different Task IDs', fontsize=40)
#add lables
plt.gca().set(ylabel = "Cost")
plt.gca().set(title = "Suppliers’ Costs for Different Task IDs",
              xlabel = "Task ID",
              ylabel = "Cost")


# Added plot (Boxplot for supplier and cost)

#decide the  figure size and autolayout
plt.rcParams["figure.figsize"] = [50, 20]
plt.rcParams["figure.autolayout"] = True
#create a boxplot for cost by supplier
cleaned_cost.boxplot(by="Supplier ID", showmeans=True, patch_artist=True)
plt.show()


## Part 2.5: Saving Dataframe for ML

#save the prepared supplier dataframe under this name 
final_cost = cleaned_cost
final_cost.head()
final_cost.info()

#save the prepared supplier dataframe under this name 
final_suppliers = cleaned_suppliers
final_suppliers.head()
final_suppliers.info()


#save the prepared tasks dataframe under this name 
final_tasks = cleaned_tasks
final_tasks.head()
final_tasks.info()


### Part 3: ML Model fitting and scoring
## Part 3.1 Merging Columns

#retireving the columns based on the results above
filtered_suppliers_factor = list(final_suppliers.columns[1:])
filtered_tasks_factor = list(final_tasks.columns[1:])

suppliers_factor_list = ["Supplier ID"] + filtered_suppliers_factor
tasks_factor_list = ["Task ID"] + filtered_tasks_factor


#merging the dataframe cost and tasks
final_df = final_cost.merge(final_tasks, how = "left", on = ["Task ID"])
print(final_df.shape)
final_df.head()

#merging the dataframe new_df and suppliers 
final_df = final_df.merge(final_suppliers, how = "left", on = ["Supplier ID"])
print(final_df.shape)
final_df.head()



## Part 3.2: Train-Test-Split

#retrieving a unique list of all the task smaples 
Groups = final_df["Task ID"].unique()
print(len(Groups))
print(Groups)

#generating a set of indexes that would be in the test set randomly 
random.seed(seed)
test_index = random.sample(range(0,len(Groups)),20)
print(test_index)

#select these task_id based on the index 
TestGroups = Groups[test_index]
print(TestGroups)


#retrieving the training set 
train = final_df[~final_df["Task ID"].isin(TestGroups)]
print(train.shape)
train.head()


#retrieving the testing set
test = final_df[final_df["Task ID"].isin(TestGroups)]
print(test.shape)
print(test["Task ID"].unique())
test.head()


X_train = train[filtered_suppliers_factor + filtered_tasks_factor]
print(X_train.shape)
X_train.head()


X_test = test[filtered_suppliers_factor + filtered_tasks_factor]
print(X_test.shape)
X_test.head()

y_train = train["Cost"]
print(y_train.shape)
y_train.head()

y_test = test["Cost"]
print(y_test.shape)
y_test.head()


## Part 3.3: ML model testing

#fitting the RidgeRegression model
ridge_reg_base = Ridge(random_state = seed)
ridge_reg_base.fit(X_train, y_train)

#checking on the base params
ridge_reg_base.get_params()

#checking the intercept terms 
ridge_reg_base.intercept_

#checking the coefficient terms
ridge_reg_base.coef_

#checking the score on train set 
ridge_reg_base.score(X_train, y_train)

#checking the score on test set 
ridge_reg_base.score(X_test, y_test)


## Part 3.4: Checking based on self defined error functions

#self defined function to retrieve the chosen supplier, error and rmse given a data set and the prediction 
def self_defined_rmse_v3(data, y_pred):
    data["pred_cost"] = y_pred
    result_df_groupped = data.groupby("Task ID")["Cost", "pred_cost"]
    errors = []
    suppliers = []
    task_id = []
    
    #loppin through the groupby object (by task id) 
    for index, data in result_df_groupped:
        cost = data["Cost"].values
        pred = data["pred_cost"].values
        #computing error 
        real_cost = np.min(cost)
        pred_cost = cost[np.argmin(pred)]
        errors.append(real_cost - pred_cost)
        #saving task id and selected suppliers
        task_id.append(index)
        suppliers.append(np.argmin(pred))
    #computing rmse 
    final_rmse = ((np.array(errors)**2).sum()/len(errors))**0.5
    return(errors, suppliers, task_id, final_rmse)


#a helper function to validate the model on train and test set given the model 
def validate_model(model, train, test, X_train, X_test):
    print("Validation on Train Set")
    y_pred = model.predict(X_train)
    y_pred.shape

    data = train.copy()
    #calling function from above 
    errors, suppliers, task_id, final_rmse = self_defined_rmse_v3(data, y_pred)
    print(f"The final rmse {final_rmse}")
    print(f"\nTask IDs")
    print(task_id)
    print(f"\nChosen Supppliers")
    print(suppliers)
    print(f"\nErrors")
    print(errors)
    
    df_train = pd.DataFrame({"Train_Test":["train"]*len(task_id),
                            "Task_ID": task_id, 
                            "Selected_Supplier": suppliers,
                            "Errors": errors})
    
    print("\nValidation on Test Set")
    y_pred = model.predict(X_test)
    y_pred.shape

    data = test.copy()
    #calling function from above 
    errors, suppliers, task_id, final_rmse = self_defined_rmse_v3(data, y_pred)
    print(f"The final rmse {final_rmse}")
    print(f"\nTask IDs")
    print(task_id)
    print(f"\nChosen Supppliers")
    print(suppliers)
    print(f"\nErrors")
    print(errors)

    df_test = pd.DataFrame({"Train_Test":["test"]*len(task_id),
                        "Task_ID": task_id, 
                        "Selected_Supplier": suppliers,
                        "Errors": errors})
    
    #retuning the dataframe 
    df_return = pd.concat([df_train, df_test], ignore_index = True)
    
    return(df_return)


#saving the model results for the base model 
ridge_reg_base_results = validate_model(ridge_reg_base, train, test, X_train, X_test)
ridge_reg_base_results.to_csv("./ridge_reg_base_results.csv", index = False)


### Part 4: Cross-validation

tasks_id = train["Task ID"].astype(str)
tasks_id

#customized scoring function 
def customized_error_v2(y, y_pred):
    #converting array to 1d format 
    y_orig = np.array(y).ravel()
    y_pred = np.array(y_pred).ravel()
    
    #compuring the actual minimum in the test set 
    orig = np.min(y_orig)
                               
    #retrieving the cost of the minimum index based on the predicted cost 
    pred = y_orig[np.argmin(y_pred)]
                               
    error = orig-pred #error as the original - machine learning prediction 
    return(error)

#creating a self defined scorer
error_func_v2 = make_scorer(customized_error_v2)


#initialzing the empty instance of 
ridge_reg_cv = Ridge(random_state = seed)

#initializing the cross validation method 
logo = LeaveOneGroupOut()

#performing the cross validation 
errors = cross_val_score(estimator = ridge_reg_cv, #specifying the regression instance
                         X = X_train, y = y_train, #specifying the x and y data frames
                         groups = tasks_id, #specifying for group
                         scoring = error_func_v2, #leveraging our own scoring method                       
                         cv = logo, #setting the cv to be LeaveOneGroup out 
                         n_jobs = -1) #utilizing all of the cpus 

#visualizing the errors
print(f"There are {len(errors)} errors in total")
print(f"the average errors are {np.mean(errors)} ")
print(f"the standard deviation of errors {np.std(errors)}")

#compuring the rmse 
rmse = ((errors**2).sum()/len(errors))**0.5
print(f"\nThe RMSE is {rmse}")

#visualizing the errors 
print("\nThe errors are:")
print(errors)

### Part 5: Hyperparameter Optimization

#helper function to retrieve results from hyperparameter tuning, computing rmse 
#and visualising rmse performance across different values
def hp_tuning_results_clean_rr(gs_cv):
    #creating an empty dataframe 
    gs_result_df_raw = pd.DataFrame()

    #retrieving the results from grid search
    n_split = gs_cv.n_splits_
    #retrieving the 
    param_combo = list(gs_cv.cv_results_["params"][0].keys())

    #extracting information from the gs_cv.cv_results_

    #for each split from the leave one out 
    for split in range(n_split):
        #for each of the parameters combinations 
        for i, param in enumerate(gs_cv.cv_results_["params"]):
            cur_entry = dict()
            #saving each of the individual parmaeter values in
            for param_name in param_combo:
                cur_entry[param_name] = param[param_name]
            #saving the current split 
            cur_entry["split"] = split
            #accessing the information (error) on the current split and the current param combo 
            cur_entry["error"] = gs_cv.cv_results_[f'split{split}_test_score'][i]
            cur_entry["squared_error"] = cur_entry["error"]**2
            #saving it as a new entry of the dataframe 
            gs_result_df_raw = gs_result_df_raw.append(pd.DataFrame(cur_entry, index = [0]), ignore_index = True)

    #groupby the parameters and compute RMSE using the n squared errors 
    gs_result_df = pd.DataFrame(gs_result_df_raw.groupby(param_combo, as_index = False)["squared_error"].sum())
    gs_result_df["RMSE"] = (gs_result_df["squared_error"]/gs_cv.n_splits_)**0.5
    gs_result_df_sort = gs_result_df.sort_values(by = ["RMSE", "alpha"], ascending = [True, True])
    
    #plot the result by solver type
    plt.figure(figsize = (15,10))
    plt.plot(gs_result_df.groupby('alpha')['alpha'].unique(),gs_result_df.groupby('alpha')["RMSE"].min(),label='best param',color='black') 

    plt.plot(gs_result_df.groupby('alpha')['alpha'].unique(),gs_result_df[gs_result_df['solver']=='auto']["RMSE"],label='auto',color='red')

    plt.plot(gs_result_df.groupby('alpha')['alpha'].unique(),gs_result_df[gs_result_df['solver']=='svd']["RMSE"],label='svd',color='blue')

    plt.plot(gs_result_df.groupby('alpha')['alpha'].unique(),gs_result_df[gs_result_df['solver']=='cholesky']["RMSE"],label='cholesky',color='orange')

    plt.plot(gs_result_df.groupby('alpha')['alpha'].unique(),gs_result_df[gs_result_df['solver']=='lsqr']["RMSE"],label='lsqr',color='yellow')

    plt.plot(gs_result_df.groupby('alpha')['alpha'].unique(),gs_result_df[gs_result_df['solver']=='sparse_cg']["RMSE"],label='sparse_cg',color='green')
    
    plt.plot(gs_result_df.groupby('alpha')['alpha'].unique(),gs_result_df[gs_result_df['solver']=='sag']["RMSE"],label='sag',color='purple')
    
    plt.plot(gs_result_df.groupby('alpha')['alpha'].unique(),gs_result_df[gs_result_df['solver']=='saga']["RMSE"],label='saga',color='cyan')

    plt.xlabel('Alpha')
    plt.ylabel('RMSE')
    plt.title('Hyper-parameter Tuning %d'%gs)
    plt.legend(fontsize=20)
    plt.show()
    return(gs_result_df_sort)

## Part 5.1: Ridge Regression Grid Search 1

#first grid search for ridge regression
#setting up parameter grid 
parameters = {'alpha':[1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5],   
             'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
             'random_state':[42]} 

# define the model/ estimator
model = Ridge()

#initializing the cross validation method 
logo = LeaveOneGroupOut()

#retrieving the groups for logo
gps = list(train["Task ID"].astype(str)) 

# define the grid search
Ridge_reg= GridSearchCV(model, parameters, #specifying the x and y data frames
                         scoring = error_func_v2, #leveraging our own scoring method                       
                         cv = logo, #setting the cv to be LeaveOneGroup out 
                         n_jobs = -1, #utilizing all of the cpus 
                        verbose = 1)

#fit the grid search
Ridge_reg.fit(X_train, y_train, groups = gps)

#finding best parameter 
print("Best parameters found:", Ridge_reg.best_params_)
print("Mean CV score of best parameters:", Ridge_reg.best_score_)


#visualising the results of grid 1 
gs=1
ridge_reg_1_cv_results = hp_tuning_results_clean_rr(Ridge_reg)
ridge_reg_1_cv_results


## Part 5.2: Ridge Regression Grid Search 2

#grid search 2 for ridge regression
#setting up parameter grid 
parameters = {'alpha':[1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000, 250000, 500000],     
             'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
             'random_state':[42]} 

# define the model/ estimator
model = Ridge()

#initializing the cross validation method 
logo = LeaveOneGroupOut()

#retrieving the groups for logo
gps = list(train["Task ID"].astype(str)) 

# define the grid search
Ridge_reg= GridSearchCV(model, parameters, #specifying the x and y data frames
                         scoring = error_func_v2, #leveraging our own scoring method                       
                         cv = logo, #setting the cv to be LeaveOneGroup out 
                         n_jobs = -1, #utilizing all of the cpus 
                        verbose = 1)

#fit the grid search
Ridge_reg.fit(X_train, y_train, groups = gps)

#finding best parameter 
print("Best parameters found:", Ridge_reg.best_params_)
print("Mean CV score of best parameters:", Ridge_reg.best_score_)


#gs2
gs=2
ridge_reg_2_cv_results = hp_tuning_results_clean_rr(Ridge_reg)
ridge_reg_2_cv_results



## Part 5.3: Validation of Tuned Ridge Regression

#refitting using selected hyperparameters
ridge_reg_tuned = Ridge(alpha= 250000, solver='lsqr',random_state = seed)
ridge_reg_tuned.fit(X_train, y_train)


#checking on the parameters
ridge_reg_tuned.get_params()


#checking the intercept terms 
ridge_reg_tuned.intercept_

#checking the intercept terms 
ridge_reg_tuned.intercept_

#checking the score on train set 
ridge_reg_tuned.score(X_train, y_train)

#checking the score on test set 
ridge_reg_tuned.score(X_test, y_test)

#saving the results 
ridge_reg_tuned_results = validate_model(ridge_reg_tuned, train, test, X_train, X_test)
ridge_reg_tuned_results.to_csv("./ridge_reg_tuned_results.csv", index = False)


### Part 6: Altenrative Model

## Part 6.1: Benchmarking Alternative Models
#Retrieve task id in the train set 

tasks_id = train['Task ID']

#evaluate each model in turn
names = []

models=[]
models.append(('LSO',Lasso()))
models.append(('MLR',LinearRegression()))
models.append(('DCT',DecisionTreeRegressor(random_state=101)))
models.append(('RFR',RandomForestRegressor(random_state=101)))
models.append(('GBR',GradientBoostingRegressor(random_state=101)))

for name, model in models :
    start = time.time()
    errors = cross_val_score(model, 
                                 X_train, y_train, 
                                 groups = tasks_id, #specifying for group
                                 scoring = error_func_v2, #leveraging our own scoring method                       
                                 cv = logo, #setting the cv to be LeaveOneGroup out 
                                 n_jobs = -1) #utilizing all of the cpus
    
    #visualizing the errors
    print("\n"+name+":")
    print(f"There are {len(errors)} errors in total")
    print(f"the average errors are {np.mean(errors)} ")
    print(f"the standard deviation of errors {np.std(errors)}")

    #compuring the rmse 
    rmse = ((errors**2).sum()/len(errors))**0.5
    print(f"The RMSE is {rmse}")
    end = time.time()
    print(f"the computational time is {end-start} ")



## Part 6.2: Base Gradient Boosting Regressor Model


gbr_base = GradientBoostingRegressor(random_state=seed)
gbr_base.fit(X_train, y_train)

gbr_base.get_params()

gbr_base.score(X_train,y_train)

gbr_base.score(X_test,y_test)


#saving the result
gbr_base_results = validate_model(gbr_base, train, test, X_train, X_test)
gbr_base_results.to_csv("./gbr_base_results.csv", index = False)

## Part 6.3: Gradient Boosting Regression Grid Search 1

def hp_tuning_results_clean_alternative(gs_cv):
    #creating an empty dataframe 
    gs_result_df_raw = pd.DataFrame()

    #retrieving the results from grid search
    n_split = gs_cv.n_splits_
    #retrieving the 
    param_combo = list(gs_cv.cv_results_["params"][0].keys())

    #extracting information from the gs_cv.cv_results_

    #for each split from the leave one out 
    for split in range(n_split):
        #for each of the parameters combinations 
        for i, param in enumerate(gs_cv.cv_results_["params"]):
            cur_entry = dict()
            #saving each of the individual parmaeter values in
            for param_name in param_combo:
                cur_entry[param_name] = param[param_name]
            #saving the current split 
            cur_entry["split"] = split
            #accessing the information (error) on the current split and the current param combo 
            cur_entry["error"] = gs_cv.cv_results_[f'split{split}_test_score'][i]
            cur_entry["squared_error"] = cur_entry["error"]**2
            #saving it as a new entry of the dataframe 
            gs_result_df_raw = gs_result_df_raw.append(pd.DataFrame(cur_entry, index = [0]), ignore_index = True)

    #groupby the parameters and compute RMSE using the n squared errors 
    gs_result_df = pd.DataFrame(gs_result_df_raw.groupby(param_combo, as_index = False)["squared_error"].sum())
    gs_result_df["RMSE"] = (gs_result_df["squared_error"]/gs_cv.n_splits_)**0.5
    #gs_result_df["std"]=gs_result_df["squared_error"] #square root it to get the standard deviation!
    gs_result_df["std"]=((gs_result_df["RMSE"]-gs_result_df["RMSE"].mean())**2/len(gs_result_df["RMSE"]))**0.5
    gs_result_df_sort = gs_result_df.sort_values(by = ["RMSE"], ascending = True)
    return(gs_result_df_sort)

#setting up parameter grid 

parameters = {
        'n_estimators': [50, 100, 250, 500],
        'learning_rate': [0.01, 0.05, 0.10,0.5],
        'max_depth': [2,4,8,16],
        'loss': ['squared_error', 'absolute_error', 'huber'],
        'random_state': [42]
        }


# define the model/ estimator
model = GradientBoostingRegressor(random_state=42)

#initializing the cross validation method 
logo = LeaveOneGroupOut()

#retrieving the groups for logo
gps = list(train["Task ID"].astype(str))

#start
start = time.time()
# define the grid search
gbr_gs_1 = GridSearchCV(model, parameters, #specifying the x and y data frames
                         scoring = error_func_v2, #leveraging our own scoring method                       
                         cv = logo, #setting the cv to be LeaveOneGroup out 
                         n_jobs = -1, #utilizing all of the cpus 
                        verbose = 3)

#fit the grid search
gbr_gs_1.fit(X_train, y_train, groups = gps)

#retrieve the result
gbr_gs_1_cv_results = hp_tuning_results_clean_alternative(gbr_gs_1)

#end
end = time.time()
print(f"the computational time is {end-start} ")

gbr_gs_1_cv_results = hp_tuning_results_clean_alternative(gbr_gs_1)
gbr_gs_1_cv_results.to_csv("./gbr_gs_1.csv")
with open('gbr_gs1_saved_dictionary.pkl', 'wb') as f:
    pickle.dump(gbr_gs_1, f)
gbr_gs_1_cv_results.head(20)


Part 6.4: Gradient Boosting Regression Grid Search 2


#setting up parameter grid 
parameters = {
        'n_estimators': [250, 375, 500, 625],
        'learning_rate': [0.025, 0.05, 0.075, 0.10, 0.125],
        'max_depth': [2,4,6,8],
        'loss': ['squared_error', 'absolute_error', 'huber'],
        'random_state': [42]
        }


# define the model/ estimator
model = GradientBoostingRegressor(random_state=42)

#initializing the cross validation method 
logo = LeaveOneGroupOut()

#retrieving the groups for logo
gps = list(train["Task ID"].astype(str))

#start
start = time.time()
# define the grid search
gbr_gs_2 = GridSearchCV(model, parameters, #specifying the x and y data frames
                         scoring = error_func_v2, #leveraging our own scoring method                       
                         cv = logo, #setting the cv to be LeaveOneGroup out 
                         n_jobs = -1, #utilizing all of the cpus 
                        verbose = 3)

#fit the grid search
gbr_gs_2.fit(X_train, y_train, groups = gps)

#retrieve the result
#gbr_gs_2_cv_results = hp_tuning_results_clean_alternative(gbr_gs_1)

#end
end = time.time()
print(f"the computational time is {end-start} ")



gbr_gs_2_cv_results = hp_tuning_results_clean_alternative(gbr_gs_2)
gbr_gs_2_cv_results.to_csv("./gbr_gs_2.csv")
with open('gbr_gs1_saved_dictionary.pkl', 'wb') as f:
    pickle.dump(gbr_gs_2, f)
gbr_gs_2_cv_results.head(20)



## Part 6.5: Validation of Tuned Gradient Boosting Regressor

#refitting using selected hyperparameters
gbr_tuned = GradientBoostingRegressor(n_estimators=625, learning_rate=0.1, max_depth=2,loss='absolute_error',
                                            random_state=seed)
gbr_tuned.fit(X_train, y_train)


#saving the result
gbr_tuned_results = validate_model(gbr_tuned, train, test, X_train, X_test)
gbr_tuned_results.to_csv("./gbr_tuned_results.csv", index = False)















