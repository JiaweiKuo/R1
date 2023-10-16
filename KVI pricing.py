# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:16:42 2023

@author: Kuo
"""

# libraries

pip install transformers torch scikit-learn


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn import preprocessing
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import torch


## load data and check
DATA = pd.read_csv("D:\BA_Dissertation\script\data_epricing.csv")

print(DATA.head(n=10))
DATA.info() # row number correct
#print(DATA.isna().sum())


Pcategory_counts = DATA['primaryCategories'].value_counts()
total_count = Pcategory_counts.sum()
category_ratios = Pcategory_counts / total_count

pieDATA = DATA.copy()


Electronics = ['Electronics']

def ElectronicsP(txt):
    if any(x in txt for x in Electronics):
        return 'Electronics'
    else:
        return 'Others'
    
pieDATA['primaryCategories'] = pieDATA['primaryCategories'].apply(ElectronicsP)

piePcategory_counts = pieDATA['primaryCategories'].value_counts()


# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(piePcategory_counts, labels=piePcategory_counts.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.

# Add a title
plt.title("Pie Chart of Primary Category Distribution")

# Show the plot
plt.show()


pc_Elec = DATA[DATA['primaryCategories'] == 'Electronics']
pc_Elec = pc_Elec.reset_index(drop=True)



pc_Elec.info() # row number correct # 14482 rows
#print(pc_Elec) 

pc_Elec['id'] = pc_Elec['id'].astype('category')
pc_Elec['name'] = pc_Elec['name'].astype('category')
pc_Elec['prices.availability'] = pc_Elec['prices.availability'].astype('category')
pc_Elec['prices.condition'] = pc_Elec['prices.condition'].astype('category')
pc_Elec['prices.merchant'] = pc_Elec['prices.merchant'].astype('category')
pc_Elec['prices.shipping'] = pc_Elec['prices.shipping'].astype('category')
pc_Elec['brand'] = pc_Elec['brand'].astype('category')
pc_Elec['prices.currency'] = pc_Elec['prices.currency'].astype('category')
pc_Elec['categories'] = pc_Elec['categories'].astype('category')
pc_Elec['manufacturer'] = pc_Elec['manufacturer'].astype('category')

pc_Elec.info()

## add and remove columns

pc_Elec['diff'] = pc_Elec['prices.amountMax'] - pc_Elec['prices.amountMin']

pc_Elec = pc_Elec.drop(columns=['prices.sourceURLs','ean','imageURLs','sourceURLs','primaryCategories','upc','weight','Unnamed: 26','Unnamed: 27','Unnamed: 28','Unnamed: 29','Unnamed: 30'])


# unify currency

pc_Elec['SeenYear'] = pc_Elec['prices.dateSeen'].str.slice(0, 4)
pc_Elec['SeenYear'] = pc_Elec['SeenYear'].astype(int)
pc_Elec['prices.currency'] = pc_Elec['prices.currency'].astype(str)


historical_average_rates = {
    2014: {'GBP': 1, 'USD': 1.647701, 'EUR': 1.240494, 'CAD': 1.818736, 'SGD': 2.087001},
    2015: {'GBP': 1, 'USD': 1.528504, 'EUR': 1.377982, 'CAD': 1.955011, 'SGD': 2.101416},
    2016: {'GBP': 1, 'USD': 1.355673, 'EUR': 1.224833, 'CAD': 1.797122, 'SGD': 1.871997},
    2017: {'GBP': 1, 'USD': 1.288611, 'EUR': 1.141317, 'CAD': 1.67142, 'SGD': 1.778214},
    2018: {'GBP': 1, 'USD': 1.334801, 'EUR': 1.130081, 'CAD': 1.729693, 'SGD': 1.799958}
}



# Function to retrieve conversion rate for a year and currency
def get_conversion_rate(year, currency):
    if year in historical_average_rates and currency in historical_average_rates[year]:
        return historical_average_rates[year][currency]
    return None

# Function to convert price to target currency using conversion rate
def convert_price(row, target_currency, column_name):
    original_currency = row['prices.currency']
    year = row['SeenYear']
    conversion_rate = get_conversion_rate(year, target_currency)
    
    if conversion_rate is None:
        return None
    
    converted_price = row[column_name] * conversion_rate
    return converted_price

# Target currency for conversion
target_currency = 'USD'

# Apply the conversion function to each price column
price_columns = ['prices.amountMax', 'prices.amountMin', 'diff']
for column_name in price_columns:
    pc_Elec[column_name] = pc_Elec.apply(lambda row: convert_price(row, target_currency, column_name), axis=1)
    
pc_Elec.info()



### barplots

#prices.availability
ava_counts = pc_Elec['prices.availability'].value_counts()
print(ava_counts)
plt.bar(ava_counts.index, ava_counts.values)

plt.xlabel('ava')
plt.ylabel('Count')
plt.title('Bar Chart of ava')
plt.xticks(rotation=90)
plt.show()

#prices.condition
cond_counts = pc_Elec['prices.condition'].value_counts()
print(cond_counts)
plt.bar(cond_counts.index, cond_counts.values)

plt.xlabel('cond')
plt.ylabel('Count')
plt.title('Bar Chart of cond')
plt.xticks(rotation=90)
plt.show()

#prices.merchant
merch_counts = pc_Elec['prices.merchant'].value_counts()
print(merch_counts)
plt.bar(merch_counts.index, merch_counts.values)

plt.xlabel('merch')
plt.ylabel('Count')
plt.title('Bar Chart of merch')
plt.xticks(rotation=90)
plt.show()


grouped_data = pc_Elec.groupby(by='prices.merchant')
mean_by_group = grouped_data['diff'].mean()
print(mean_by_group.sort_values(ascending=False).head(20)) # need boxplot to justify


##  1553 merchants in total
merchs = pc_Elec['prices.merchant'].unique()
print('In total',len(merchs),'merchants')
## print(merchs)

## 309 brands in total
brands = pc_Elec['brand'].unique()
print('In total',len(brands),'brands')
## print(brands)

## 1192 categories in total
cates = pc_Elec['categories'].unique()
print('In total',len(cates),'categories')
## print(cates)

merch_counts = pc_Elec['prices.merchant'].value_counts()
print(merch_counts.head(10))

brand_counts = pc_Elec['brand'].value_counts()
print(brand_counts.head(15))

cate_counts = pc_Elec['categories'].value_counts()
print(cate_counts.head(10))


## missing value preprocessing
pc_Elec.info()
pc_Elec.isna().sum()



pc_Elec['prices.condition'].fillna(value='New', inplace=True)
pc_Elec['prices.availability'].fillna(value='In Stock', inplace=True)

pc_Elec['prices.shipping'] = pc_Elec['prices.shipping'].astype(str)
pc_Elec['prices.shipping'].fillna(value='NOINFO', inplace=True)

pc_Elec.isna().sum()

pc_Elec.loc[pc_Elec['prices.shipping'] == 'nan', 'prices.shipping'] = 'noINFO'

print(pc_Elec['prices.shipping'].unique())

ship_counts = pc_Elec['prices.shipping'].value_counts()
print(ship_counts.head(15))






## key categories transfer

# price ava
print(pc_Elec['prices.availability'].unique())

for y in pc_Elec['prices.availability'].unique():
    print(y)

ava_true = ['Yes', 'In Stock', 'TRUE','yes','Special Order','32 available','7 available','Limited Stock']

def ava_cate(txt):
    if any(x in txt for x in ava_true):
        return 'yes'
    else:
        return 'no'
    

pc_Elec['prices.availability'] = pc_Elec['prices.availability'].apply(ava_cate)




# price cond
x = 1
for y in pc_Elec['prices.condition'].unique():
    x+=1
    print(y, '-', x)

cond_true = ['new','New','New other (see details)','Brand New']

def cond_cate(txt):
    if any(x in txt for x in cond_true):
        return 'new'
    else:
        return 'not new'
    

pc_Elec['prices.condition'] = pc_Elec['prices.condition'].apply(cond_cate)


# transfer result check
ava_counts = pc_Elec['prices.availability'].value_counts()
print(ava_counts)

cond_counts = pc_Elec['prices.condition'].value_counts()
print(cond_counts)


pc_Elec.info()


## delete the products which are not new

pc_Elec = pc_Elec.drop(pc_Elec[pc_Elec['prices.condition'] == 'not new'].index)

pc_Elec.info()




## impute merch column

# Step 1: One-hot encode categorical columns


pc_Elec_encoded = pd.get_dummies(pc_Elec, columns=['prices.availability','prices.isSale','brand','prices.shipping'], dummy_na=True)

pc_Elec_encoded.info()

# Step 2: Split the data into known and unknown (missing) parts
known_data = pc_Elec_encoded[pc_Elec_encoded['prices.merchant'].notnull()]
unknown_data = pc_Elec_encoded[pc_Elec_encoded['prices.merchant'].isnull()]

# Step 3: Define features and target
X = known_data.drop(['prices.merchant', 'prices.condition','id','prices.amountMax','prices.amountMin','prices.currency','prices.dateSeen','asins','categories','dateAdded','dateUpdated','keys','manufacturer','manufacturerNumber','name'], axis=1)
y = known_data['prices.merchant']


# Step 5: Train a machine learning model on the known data
model = RandomForestClassifier(n_estimators=100)  # You can use other classifiers as well
model.fit(X, y)

# Predict the missing values using the random forest model
X_missing = unknown_data.drop(['prices.merchant', 'prices.condition','id','prices.amountMax','prices.amountMin','prices.currency','prices.dateSeen','asins','categories','dateAdded','dateUpdated','keys','manufacturer','manufacturerNumber','name'], axis=1)
unknown_data['prices.merchant'] = model.predict(X_missing)


# Combine the two dataframes back together
pc_Elec_imputed = pd.concat([known_data, unknown_data], sort=False).sort_index()

# Print the imputed dataframe
print(pc_Elec_imputed)

pc_Elec_imputed.info()
pc_Elec_imputed.isna().sum()

r_squared = model.score(X, y)
print("R-squared value of imputed values:", r_squared) #0.8233760257688473



## integrate the dummies

pc_Elec_C = pc_Elec_imputed.copy()


non_dummy_cols = []
ava_dummies = []
brand_dummies = []
shipping_dummies = []
sale_dummies = []


for col in pc_Elec_C.columns :
    if '_' not in col:
        print(col)
        non_dummy_cols.append(col)
    if 'prices.availability_' in col:
        ava_dummies.append(col)
    
    if 'brand_' in col:
        brand_dummies.append(col)
    if 'prices.shipping_' in col:
        shipping_dummies.append(col)
    if 'prices.isSale_' in col:
        sale_dummies.append(col)    
        
        
        
pc_Elec_CD = pc_Elec_C.drop(non_dummy_cols, axis=1)     
pc_Elec_CD   
        
pc_Elec_CD = pc_Elec_CD.astype(int)


def integrate_dummy_columns(df, column_prefix, select_columns):
    cols = df[select_columns].columns.str.replace(column_prefix, '')
    df[column_prefix] = cols[df[select_columns].values.argmax(axis=1)]
    return df



# Integrate dummy columns for prices.availability
pc_Elec_CD = integrate_dummy_columns(pc_Elec_CD, 'prices.availability_', ava_dummies)

# Integrate dummy columns for prices.condition


pc_Elec_CD = integrate_dummy_columns(pc_Elec_CD, 'brand_', brand_dummies)
pc_Elec_CD = integrate_dummy_columns(pc_Elec_CD, 'prices.shipping_', shipping_dummies)
pc_Elec_CD = integrate_dummy_columns(pc_Elec_CD, 'prices.isSale_', sale_dummies)


print(pc_Elec_CD)


for col in pc_Elec_CD.columns :
    if col[-1] ==  '_':
        print(col)
        
# the additional '_' in column names need to be deleted

## concat the dummy integrated one and the other columns

other_cols = ['id', 'prices.condition','prices.amountMax','prices.amountMin','prices.currency','prices.dateSeen','prices.merchant','asins','categories','dateAdded','dateUpdated','keys','manufacturer','manufacturerNumber','name','diff']
integ_cols = ['prices.availability_','brand_','prices.shipping_','prices.isSale_']
pc_Elec_1 = pc_Elec_imputed.loc[:,other_cols]
pc_Elec_2 = pc_Elec_CD.loc[:,integ_cols]


pc_Elec_Fnl = pd.concat([pc_Elec_1, pc_Elec_2], axis=1)

pc_Elec_Fnl.info()

pc_Elec_Fnl.rename(columns={'prices.availability_': 'prices.availability'}, inplace=True)

pc_Elec_Fnl.rename(columns={'brand_': 'brand'}, inplace=True)
pc_Elec_Fnl.rename(columns={'prices.shipping_': 'prices.shipping'}, inplace=True)
pc_Elec_Fnl.rename(columns={'prices.isSale_': 'prices.isSale'}, inplace=True)

pc_Elec_Fnl.info()

pc_Elec_Fnl.isna().sum()

# check again the data
pc_Elec_Fnl.to_csv('D:/BA_Dissertation/script/pc_Elec_Fnl.csv', index=False)





## category identify session

# Load the pre-trained BERT model and tokenizer

p_name = pc_Elec_Fnl['name']
p_name.astype(str)

product_names = p_name  # List of product names

model_name = 'bert-base-uncased'


# Tokenization using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_names = [tokenizer.encode(name, add_special_tokens=True) for name in product_names]

# Load pre-trained BERT model for embeddings
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Obtain embeddings from BERT for each product name
embeddings = []
for tokens in tokenized_names:
    with torch.no_grad():
        outputs = model(input_ids=torch.tensor([tokens]))
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        embeddings.append(embedding)
embeddings = np.concatenate(embeddings, axis=0)

# Apply PCA for dimensionality reduction (optional)
# pca = PCA(n_components=50)
# reduced_embeddings = pca.fit_transform(embeddings)

# Apply K-Means clustering
num_clusters = 10  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters)
cluster_ids = kmeans.fit_predict(embeddings)

# Analyze clusters and assign categories
# Based on your analysis, assign category labels to clusters
category_labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

# Assign categories to product names based on cluster IDs
product_categories = [category_labels[cluster_id] for cluster_id in cluster_ids]

# Create a new DataFrame with product names and categories
data = {'Product Name': product_names, 'Category': product_categories}
dff = pd.DataFrame(data)

dff # only the product name column and raw category column

# Save the DataFrame to a file or perform further analysis

dff.to_csv('D:/BA_Dissertation/script/cateTestraw.csv', index=False)



cat_counts = dff['Category'].value_counts()
print(cat_counts.head(10))


## load back the modified cateTest file, combine, and check

mod_cate = pd.read_csv("D:\BA_Dissertation\script\cateTest.csv")


mod_cate.rename(columns={'Category': 'CATEGORY', 'Product Name': 'PN'}, inplace=True)


pc_Elec_Fnl2 = pc_Elec_Fnl.copy()


pc_Elec_Fnl2 = pc_Elec_Fnl2.reset_index(drop=True)

pc_Elec_Fnl2['CATEGORY'] = mod_cate['CATEGORY']
pc_Elec_Fnl2['PN'] = mod_cate['PN']


# Check if the content of two columns is identical
are_identical = (pc_Elec_Fnl2['name'] == pc_Elec_Fnl2['PN']).all()

if are_identical:
    print("The content of the columns is identical.")
else:
    print("The content of the columns is not identical.")
    
pc_Elec_Fnl2.isna().sum()


columns_to_dropp = ['PN', 'manufacturer','manufacturerNumber'] # don't need the columns for further analysis
pc_Elec_Fnl2 = pc_Elec_Fnl2.drop(columns=columns_to_dropp)


## give the CATEGORIES name

pc_Elec_Fnl2['CATEGORY'] = pc_Elec_Fnl2['CATEGORY'].astype(str)

# Define a dictionary for value replacement
value_mapping = {'5': 'Drive_Memory', '8': 'Headphone_Speaker', '3': 'TV', '7':'StereoSystem','1':'Laptop', '0':'Monitor','4':'Camera','9':'Others','2':'Projector','6':'MobilePhone'}

# Replace values in the 'column1' based on the dictionary
pc_Elec_Fnl2['CATEGORY'] = pc_Elec_Fnl2['CATEGORY'].apply(lambda x: value_mapping.get(x, x))




# keyboard & mouse category adding 
condition_words = ['keyboard', 'mouse','Keyboard','Mouse','Chroma V2','Razer Mamba Chroma','Souris Wireless For Surface Windows 10','Logitech 920008149 K780 Wrls Multi Device Keyb']
new_category = 'Keyboard_Mouse'

for word in condition_words:
    pc_Elec_Fnl2.loc[pc_Elec_Fnl2['name'].str.contains(word), 'CATEGORY'] = new_category




newcate_counts = pc_Elec_Fnl2['CATEGORY'].value_counts()
print(newcate_counts)


## further preprocessing

jbl_combine = ['JBL', 'Jbl']
new_jbl = 'JBL'

pc_Elec_Fnl2.loc[pc_Elec_Fnl2['brand'].isin(jbl_combine), 'brand'] = new_jbl


Corsair_combine = ['Corsair', 'CORSAIR']
new_Corsair = 'Corsair'

pc_Elec_Fnl2.loc[pc_Elec_Fnl2['brand'].isin(Corsair_combine), 'brand'] = new_Corsair


LG_combine = ['LG', 'Lg']
new_LG = 'LG'

pc_Elec_Fnl2.loc[pc_Elec_Fnl2['brand'].isin(LG_combine), 'brand'] = new_LG



bestbuy_combine = ['Bestbuy.com', 'Best Buy']
new_bestbuy = 'Bestbuy.com'

pc_Elec_Fnl2.loc[pc_Elec_Fnl2['prices.merchant'].isin(bestbuy_combine), 'prices.merchant'] = new_bestbuy


# delete 'detail about' texts


pc_Elec_Fnl2['name'] = pc_Elec_Fnl2['name'].str.replace('Details About ', '', regex=True)


# delete 'refurnbished' and 'Pre-owned' ones
texts_to_remove = ['Pre-Owned', 'Refurbished']

filtered_df = pc_Elec_Fnl2[~pc_Elec_Fnl2['name'].str.contains('|'.join(texts_to_remove))]

pc_Elec_Fnl2 = filtered_df.reset_index(drop=True)


# create simplified date columns

num_characters = 10

# Custom function to extract the first few characters
def extract_first_letters(s):
    return s[:num_characters]

# Apply the function to create a new column
pc_Elec_Fnl2['SdateUpdated'] = pc_Elec_Fnl2['dateUpdated'].apply(extract_first_letters)
pc_Elec_Fnl2['SdateAdded'] = pc_Elec_Fnl2['dateAdded'].apply(extract_first_letters)

pc_Elec_Fnl2['SdateUpdated'] = pd.to_datetime(pc_Elec_Fnl2['SdateUpdated'])
pc_Elec_Fnl2['SdateAdded'] = pd.to_datetime(pc_Elec_Fnl2['SdateAdded'])


time_period_updt = pc_Elec_Fnl2['SdateUpdated'].max() - pc_Elec_Fnl2['SdateUpdated'].min()
time_period_add = pc_Elec_Fnl2['SdateAdded'].max() - pc_Elec_Fnl2['SdateAdded'].min()

print('updated:',"Time period-", time_period_updt,'max-',pc_Elec_Fnl2['SdateUpdated'].max(),'min-',pc_Elec_Fnl2['SdateUpdated'].min())
print('added:',"Time period-", time_period_add,'max-',pc_Elec_Fnl2['SdateAdded'].max(),'min-',pc_Elec_Fnl2['SdateAdded'].min())


pc_Elec_Fnl2.info()

# export the final version of processed dataframe
pc_Elec_Fnl2.to_csv('D:/BA_Dissertation/script/pc_Elec_Fnl2.csv', index=False)


unique_cat_values = pc_Elec_Fnl2['categories'].unique()

unique_cat_df = pd.DataFrame({'Unique_Values': unique_cat_values})
unique_cat_df.to_csv('D:/BA_Dissertation/script/originalcates.csv', index=False)

### data exploring session

## General scale

BRAND_counts = pc_Elec_Fnl2['brand'].value_counts()
print(BRAND_counts.head(15))

MERCH_counts = pc_Elec_Fnl2['prices.merchant'].value_counts()
print(MERCH_counts.head(15))


uni_product = pc_Elec_Fnl2.copy()
uni_product.drop_duplicates(subset='name', keep='first', inplace=True)

pc_Elec_Fnl3 = pc_Elec_Fnl2.copy()

uni_product.info()

uniBRAND_counts = uni_product['brand'].value_counts()
print(uniBRAND_counts.head(15))

uniMERCH_counts = uni_product['prices.merchant'].value_counts()
print(uniMERCH_counts.head(15))

# brand vs cate

main_brands = ['Sony','Samsung','Apple','Pioneer','Yamaha','SanDisk','LG','Canon','Logitech','Corsair','WD','Nikon']

main_brands_df = pc_Elec_Fnl2[pc_Elec_Fnl2['brand'].isin(main_brands)]

brand_category_counts = main_brands_df.groupby(['brand', 'CATEGORY'])['name'].count().unstack().fillna(0)
brand_category_percentage = brand_category_counts.div(brand_category_counts.sum(axis=1), axis=0) * 100

ax = brand_category_percentage.plot(kind='bar', stacked=True, figsize=(10, 6))
ax.set_ylabel('Percentage')
ax.set_xlabel('Brand')
ax.set_title('Stacked Bar Plot of Category Distribution by Brand')
plt.legend(title='Category')
plt.show()



# cate vs merch

main_merchs = ['Bestbuy.com','bhphotovideo.com','Walmart.com','Amazon.com','Beach Camera','AMI Ventures Inc']

main_merchs_df = pc_Elec_Fnl2[pc_Elec_Fnl2['prices.merchant'].isin(main_merchs)]


category_merch_counts = main_merchs_df.groupby(['CATEGORY', 'prices.merchant'])['name'].count().unstack().fillna(0)
category_merch_percentage = category_merch_counts.div(category_merch_counts.sum(axis=1), axis=0) * 100

ax = category_merch_percentage.plot(kind='bar', stacked=True, figsize=(10, 6))
ax.set_ylabel('Percentage')
ax.set_xlabel('Category')
ax.set_title('Stacked Bar Plot of Merch Distribution by Category')
plt.legend(title='Merch')
plt.show()


# brand vs merch

main_brands_merch_df = main_brands_df[main_brands_df['prices.merchant'].isin(main_merchs)]


brand_merch_counts = main_brands_merch_df.groupby(['brand', 'prices.merchant'])['name'].count().unstack().fillna(0)
brand_merch_percentage = brand_merch_counts.div(brand_merch_counts.sum(axis=1), axis=0) * 100

ax = brand_merch_percentage.plot(kind='bar', stacked=True, figsize=(10, 6))
ax.set_ylabel('Percentage')
ax.set_xlabel('Brand')
ax.set_title('Stacked Bar Plot of Merch Distribution by Brand')
plt.legend(title='Merch')
plt.show()


## look into each category

cat_Headphone_Speaker = pc_Elec_Fnl2[pc_Elec_Fnl2['CATEGORY'] == 'Headphone_Speaker']
cat_Drive_Memory = pc_Elec_Fnl2[pc_Elec_Fnl2['CATEGORY'] == 'Drive_Memory']
cat_TV = pc_Elec_Fnl2[pc_Elec_Fnl2['CATEGORY'] == 'TV']
cat_StereoSystem = pc_Elec_Fnl2[pc_Elec_Fnl2['CATEGORY'] == 'StereoSystem']
cat_Camera = pc_Elec_Fnl2[pc_Elec_Fnl2['CATEGORY'] == 'Camera']
cat_Laptop = pc_Elec_Fnl2[pc_Elec_Fnl2['CATEGORY'] == 'Laptop']
cat_Keyboard_Mouse = pc_Elec_Fnl2[pc_Elec_Fnl2['CATEGORY'] == 'Keyboard_Mouse']


## main competitors in the category

cat_Headphone_Speaker.info()

hs_counts = cat_Headphone_Speaker['brand'].value_counts()
print(hs_counts.head(10)) # including dulplicate products


# Drop duplicate products in the specified column, keeping the first occurrence
uniname_Headphone_Speaker = cat_Headphone_Speaker.copy()

uniname_Headphone_Speaker.drop_duplicates(subset='name', keep='first', inplace=True)

hs_counts2 = uniname_Headphone_Speaker['brand'].value_counts()
print(hs_counts2.head(10))  # count with unique products
# assume: more kind of products & more pricing data, more competitive




dm_counts = cat_Drive_Memory['brand'].value_counts()
print(dm_counts.head(10)) # including dulplicate products


# Drop duplicate products in the specified column, keeping the first occurrence
uniname_Drive_Memory = cat_Drive_Memory.copy()

uniname_Drive_Memory.drop_duplicates(subset='name', keep='first', inplace=True)

dm_counts2 = uniname_Drive_Memory['brand'].value_counts()
print(dm_counts2.head(10))  # count with unique products






tv_counts = cat_TV['brand'].value_counts()
print(tv_counts.head(10)) # including dulplicate products


# Drop duplicate products in the specified column, keeping the first occurrence
uniname_TV = cat_TV.copy()

uniname_TV.drop_duplicates(subset='name', keep='first', inplace=True)

tv_counts2 = uniname_TV['brand'].value_counts()
print(tv_counts2.head(10))  # count with unique products






ss_counts = cat_StereoSystem['brand'].value_counts()
print(ss_counts.head(10)) # including dulplicate products


# Drop duplicate products in the specified column, keeping the first occurrence
uniname_StereoSystem = cat_StereoSystem.copy()

uniname_StereoSystem.drop_duplicates(subset='name', keep='first', inplace=True)

ss_counts2 = uniname_StereoSystem['brand'].value_counts()
print(ss_counts2.head(10))  # count with unique products





cam_counts = cat_Camera['brand'].value_counts()
print(cam_counts.head(10)) # including dulplicate products


# Drop duplicate products in the specified column, keeping the first occurrence
uniname_Camera = cat_Camera.copy()

uniname_Camera.drop_duplicates(subset='name', keep='first', inplace=True)

cam_counts2 = uniname_Camera['brand'].value_counts()
print(cam_counts2.head(10))  # count with unique products





lt_counts = cat_Laptop['brand'].value_counts()
print(lt_counts.head(10)) # including dulplicate products


# Drop duplicate products in the specified column, keeping the first occurrence
uniname_Laptop = cat_Laptop.copy()

uniname_Laptop.drop_duplicates(subset='name', keep='first', inplace=True)

lt_counts2 = uniname_Laptop['brand'].value_counts()
print(lt_counts2.head(10))  # count with unique products




km_counts = cat_Keyboard_Mouse['brand'].value_counts()
print(km_counts.head(10)) # including dulplicate products


# Drop duplicate products in the specified column, keeping the first occurrence
uniname_Keyboard_Mouse = cat_Keyboard_Mouse.copy()

uniname_Keyboard_Mouse.drop_duplicates(subset='name', keep='first', inplace=True)

km_counts2 = uniname_Keyboard_Mouse['brand'].value_counts()
print(km_counts2.head(10))  # count with unique products



## the distribution of price (diff, max, min)

cat_Headphone_Speaker.info()

cat_Headphone_Speaker.boxplot(column=['prices.amountMax','prices.amountMin'])

#plt.ylim(0, 5000)

plt.show

plt.boxplot(cat_Headphone_Speaker['diff'])
plt.show




cat_Drive_Memory.boxplot(column=['prices.amountMax','prices.amountMin'])

#plt.ylim(0, 5000)

plt.show

plt.boxplot(cat_Drive_Memory['diff'])
plt.show




cat_TV.boxplot(column=['prices.amountMax','prices.amountMin'])

#plt.ylim(0, 5000)

plt.show

plt.boxplot(cat_TV['diff'])
plt.show



cat_StereoSystem.boxplot(column=['prices.amountMax','prices.amountMin'])

#plt.ylim(0, 5000)

plt.show

plt.boxplot(cat_StereoSystem['diff'])
plt.show


cat_Camera.boxplot(column=['prices.amountMax','prices.amountMin'])

#plt.ylim(0, 5000)

plt.show

plt.boxplot(cat_Camera['diff'])
plt.show


cat_Laptop.boxplot(column=['prices.amountMax','prices.amountMin'])

#plt.ylim(0, 5000)

plt.show

plt.boxplot(cat_Laptop['diff'])
plt.show



cat_Keyboard_Mouse.boxplot(column=['prices.amountMax','prices.amountMin'])

#plt.ylim(0, 5000)

plt.show

plt.boxplot(cat_Keyboard_Mouse['diff'])
plt.show






### KVI model1 -> index score assigning

# first create a dataset with no dulpicate product

uni_product = pc_Elec_Fnl2.copy()
uni_product.drop_duplicates(subset='name', keep='first', inplace=True)

pc_Elec_Fnl3 = pc_Elec_Fnl2.copy()

# add a column for KVI index score

#uni_product['KVI_score'] = None

#uni_product.info() # 1215 unique products in total


## Criteria 1: number of brands having this category # -> (number of competitors carrying the product)

uni_product['NBC'] = uni_product.groupby('CATEGORY')['brand'].transform('nunique')



uni_product2 = uni_product.copy()

uni_product2 = uni_product2[['name','NBC']]

merged_df1 = pc_Elec_Fnl3.merge(uni_product2, on='name', how='left')

merged_df1



## Criteria 2: number of merch selling this product -> the popularity of the product -> traffic or basket driver

merchcount_product = pc_Elec_Fnl2.copy()
merchcount_product2 = uni_product.copy()

platform_counts = merchcount_product.groupby('name')['prices.merchant'].nunique().reset_index()
platform_counts.rename(columns={'prices.merchant': 'Merch Count'}, inplace=True)


merged_df = merchcount_product2.merge(platform_counts, on='name', how='left')


merged_df = merged_df[['name','Merch Count']]

merged_df2 = merged_df1.merge(merged_df, on='name', how='left')



# mean standard calculation
mstdf = merged_df2.copy()

mstdf.drop_duplicates(subset='name', keep='first', inplace=True)
mstdf

merchmean_by_category = mstdf.groupby('CATEGORY')['Merch Count'].mean()

merchmean_by_category = pd.DataFrame({'MCM': merchmean_by_category})

merged_df2 = merged_df2.merge(merchmean_by_category, on='CATEGORY', how='left')

merged_df2

## Criteria 3: the 'price difference' of the product # -> (price perception sensitivity)
# based on the diff column


# mean standard calculation: the mean value of the same products
# 'Mean diff' column



## Criteria 4: the mean of 'price difference' of the same product names # -> 'usually' (price perception sensitivity)

mean_scores = merged_df2.groupby('name')['diff'].mean().reset_index()
mean_scores.rename(columns={'diff': 'Mean diff'}, inplace=True)



merged_df2 = merged_df2.merge(mean_scores, on='name', how='left')

# mean standard calculation: the mean value of the category
diff_by_category = merged_df2.groupby('CATEGORY')['diff'].mean()

print(diff_by_category)


diff_by_category = pd.DataFrame({'diffC': diff_by_category})

merged_df2 = merged_df2.merge(diff_by_category, on='CATEGORY', how='left')

merged_df2


## Criteria 5: the product is having sale or not # -> (price perception sensitivity)
# check the sales column


## Criteria 7: shipping?




merged_df2.info()


## score assigning

## KVI model 1

merged_df2['KVI'] = 0

#unique_ship = merged_df2['prices.shipping'].unique()
#print(unique_ship)

freeshippinglst = ['Free Shipping','Free Expedited Shipping','FREE','Free Shipping for this Item','Free Delivery','Free Standard Shipping','Free Next Day Delivery (USA)','FREE Shipping.','FREE Shipping','free','Free','Free 2-Day Shipping']

for index, row in merged_df2.iterrows():
    score = 0
    if row['Merch Count'] >= row['MCM']:
        score += 2
    if row['diff'] > 0:
        score += 1
    if (row['diff'] >= row['Mean diff']) :
        score += 1
    if row['Mean diff'] >= row['diffC']:
        score += 1
    if (row['prices.isSale'] == 'TRUE') or (row['prices.shipping'] in freeshippinglst):  
        score += 1
    merged_df2.at[index, 'KVI'] = row['KVI'] + score



## KVI score check in general scale
plt.boxplot(merged_df2['KVI'])
plt.show

mean_value = merged_df2['KVI'].mean()
std_deviation = merged_df2['KVI'].std()
minimum = merged_df2['KVI'].min()
maximum = merged_df2['KVI'].max()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)
print("Minimum:", minimum)
print("Maximum:", maximum)


merged_df2.to_csv('D:/BA_Dissertation/script/merged_df2.csv', index=False)


## KVI score check for category
# headphone& speaker
kate_Headphone_Speaker = merged_df2[merged_df2['CATEGORY'] == 'Headphone_Speaker']

plt.boxplot(kate_Headphone_Speaker['KVI'])
plt.show

mean_value = kate_Headphone_Speaker['KVI'].mean()
std_deviation = kate_Headphone_Speaker['KVI'].std()
minimum = kate_Headphone_Speaker['KVI'].min()
maximum = kate_Headphone_Speaker['KVI'].max()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)
print("Minimum:", minimum)
print("Maximum:", maximum)


meanmax_value_Sony_5 = kate_Headphone_Speaker[(kate_Headphone_Speaker['brand'] == 'Sony') & (kate_Headphone_Speaker['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Sony_5 = kate_Headphone_Speaker[(kate_Headphone_Speaker['brand'] == 'Sony') & (kate_Headphone_Speaker['KVI'] == 5)]['prices.amountMin'].mean()
stdmax_value_Sony_5 = kate_Headphone_Speaker[(kate_Headphone_Speaker['brand'] == 'Sony') & (kate_Headphone_Speaker['KVI'] == 5)]['prices.amountMax'].std()
stdmin_value_Sony_5 = kate_Headphone_Speaker[(kate_Headphone_Speaker['brand'] == 'Sony') & (kate_Headphone_Speaker['KVI'] == 5)]['prices.amountMin'].std()

print("meanmax_value_Sony_5:", meanmax_value_Sony_5)
print("meanmin_value_Sony_5:", meanmin_value_Sony_5)
print("stdmax_value_Sony_5:", stdmax_value_Sony_5)
print("stdmin_value_Sony_5:", stdmin_value_Sony_5)


meanmax_value_Sennheiser_5 = kate_Headphone_Speaker[(kate_Headphone_Speaker['brand'] == 'Sennheiser') & (kate_Headphone_Speaker['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Sennheiser_5 = kate_Headphone_Speaker[(kate_Headphone_Speaker['brand'] == 'Sennheiser') & (kate_Headphone_Speaker['KVI'] == 5)]['prices.amountMin'].mean()


print("meanmax_value_Sennheiser_5:", meanmax_value_Sennheiser_5)
print("meanmin_value_Sennheiser_5:", meanmin_value_Sennheiser_5)

meanmax_value_JBL_5 = kate_Headphone_Speaker[(kate_Headphone_Speaker['brand'] == 'JBL') & (kate_Headphone_Speaker['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_JBL_5 = kate_Headphone_Speaker[(kate_Headphone_Speaker['brand'] == 'JBL') & (kate_Headphone_Speaker['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_JBL_5:", meanmax_value_JBL_5)
print("meanmin_value_JBL_5:", meanmin_value_JBL_5)



hsmaxk = kate_Headphone_Speaker['prices.amountMax'].mean()
hsmink = kate_Headphone_Speaker['prices.amountMin'].mean()

hstdx = kate_Headphone_Speaker['prices.amountMax'].std()
hstdn = kate_Headphone_Speaker['prices.amountMin'].std()

#medhs = kate_Headphone_Speaker['prices.amountMax'].median()

print(hsmaxk)
print(hsmink)

print(hstdx)
print(hstdn)

# normalization
N_Headphone_Speaker = kate_Headphone_Speaker.copy()

scaler = MinMaxScaler()
N_Headphone_Speaker['prices.amountMax'] = scaler.fit_transform(N_Headphone_Speaker[['prices.amountMax']])
N_Headphone_Speaker['prices.amountMin'] = scaler.fit_transform(N_Headphone_Speaker[['prices.amountMin']])


# remove outlier
Q1_max = N_Headphone_Speaker['prices.amountMax'].quantile(0.25)
Q3_max = N_Headphone_Speaker['prices.amountMax'].quantile(0.75)
IQR_max = Q3_max - Q1_max
lower_bound_max = Q1_max - 1.5 * IQR_max
upper_bound_max = Q3_max + 1.5 * IQR_max
outliers_max = (N_Headphone_Speaker['prices.amountMax'] < lower_bound_max) | (N_Headphone_Speaker['prices.amountMax'] > upper_bound_max)


num_outliers_max = outliers_max.sum()
print("Number of outliers:", num_outliers_max)

total_data_points = len(N_Headphone_Speaker)
percentage_outliers_max = (num_outliers_max / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_max, "%")


Q1_min = N_Headphone_Speaker['prices.amountMin'].quantile(0.25)
Q3_min = N_Headphone_Speaker['prices.amountMin'].quantile(0.75)
IQR_min = Q3_min - Q1_min
lower_bound_min = Q1_min - 1.5 * IQR_min
upper_bound_min = Q3_min + 1.5 * IQR_min
outliers_min = (N_Headphone_Speaker['prices.amountMin'] < lower_bound_min) | (N_Headphone_Speaker['prices.amountMin'] > upper_bound_min)


num_outliers_min = outliers_min.sum()
print("Number of outliers:", num_outliers_min)

total_data_points = len(N_Headphone_Speaker)
percentage_outliers_min = (num_outliers_min / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_min, "%")




combined_outliers = outliers_max | outliers_min

N_Headphone_Speaker_no_outliers = N_Headphone_Speaker[~combined_outliers]

plt.boxplot(N_Headphone_Speaker_no_outliers['prices.amountMax'])
plt.show

#nhs_counts = N_Headphone_Speaker_no_outliers['brand'].value_counts()
#print(nhs_counts)


# check and compare mean values of main competitors in a general scale

meanmax_Sony_5 = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'Sony') & (N_Headphone_Speaker_no_outliers['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_Sony_5 = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'Sony') & (N_Headphone_Speaker_no_outliers['KVI'] == 5)]['prices.amountMin'].mean()

stdmax_Sony_5 = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'Sony') & (N_Headphone_Speaker_no_outliers['KVI'] == 5)]['prices.amountMax'].std()
stdmin_Sony_5 = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'Sony') & (N_Headphone_Speaker_no_outliers['KVI'] == 5)]['prices.amountMin'].std()

print("stdmax_Sony_5:", stdmax_Sony_5)
print("stdmin_Sony_5:", stdmin_Sony_5)
print("meanmax_Sony_5:", meanmax_Sony_5)
print("meanmin_Sony_5:", meanmin_Sony_5)


meanmax_JBL_5 = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'JBL') & (N_Headphone_Speaker_no_outliers['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_JBL_5 = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'JBL') & (N_Headphone_Speaker_no_outliers['KVI'] == 5)]['prices.amountMin'].mean()
stdmax_JBL_5 = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'JBL') & (N_Headphone_Speaker_no_outliers['KVI'] == 5)]['prices.amountMax'].std()
stdmin_JBL_5 = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'JBL') & (N_Headphone_Speaker_no_outliers['KVI'] == 5)]['prices.amountMin'].std()

print("stdmax_JBL_5:", stdmax_JBL_5)
print("stdmin_JBL_5:", stdmin_JBL_5)
print("meanmax_JBL_5:", meanmax_JBL_5)
print("meanmax_JBL_5:", meanmin_JBL_5)

meanmax_Sennheiser_5 = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'Sennheiser') & (N_Headphone_Speaker_no_outliers['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_Sennheiser_5 = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'Sennheiser') & (N_Headphone_Speaker_no_outliers['KVI'] == 5)]['prices.amountMin'].mean()
stdmax_Sennheiser_5 = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'Sennheiser') & (N_Headphone_Speaker_no_outliers['KVI'] == 5)]['prices.amountMax'].std()
stdmin_Sennheiser_5 = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'Sennheiser') & (N_Headphone_Speaker_no_outliers['KVI'] == 5)]['prices.amountMin'].std()

print("stdmax_Sennheiser_5:", stdmax_Sennheiser_5)
print("stdmin_Sennheiser_5:", stdmin_Sennheiser_5)
print("meanmax_Sennheiser_5:", meanmax_Sennheiser_5)
print("meanmax_Sennheiser_5:", meanmin_Sennheiser_5)

hsmaxmean = N_Headphone_Speaker_no_outliers['prices.amountMax'].mean()
hsminmean = N_Headphone_Speaker_no_outliers['prices.amountMin'].mean()

hsmaxstd = N_Headphone_Speaker_no_outliers['prices.amountMax'].std()
hsminstd = N_Headphone_Speaker_no_outliers['prices.amountMin'].std()
print("stdmax", hsmaxstd)
print("stdmin", hsminstd)
print("meanmax", hsmaxmean)
print("meanmin", hsminmean)



# Drive& memory
kate_drive_memory = merged_df2[merged_df2['CATEGORY'] == 'Drive_Memory']

plt.boxplot(kate_drive_memory['KVI'])
plt.show

mean_value = kate_drive_memory['KVI'].mean()
std_deviation = kate_drive_memory['KVI'].std()
minimum = kate_drive_memory['KVI'].min()
maximum = kate_drive_memory['KVI'].max()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)
print("Minimum:", minimum)
print("Maximum:", maximum)


meanmax_value_SanDisk_5 = kate_drive_memory[(kate_drive_memory['brand'] == 'SanDisk') & (kate_drive_memory['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_SanDisk_5 = kate_drive_memory[(kate_drive_memory['brand'] == 'SanDisk') & (kate_drive_memory['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_SanDisk_5:", meanmax_value_SanDisk_5)
print("meanmax_value_SanDisk_5:", meanmin_value_SanDisk_5)


meanmax_value_WD_5 = kate_drive_memory[(kate_drive_memory['brand'] == 'WD') & (kate_drive_memory['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_WD_5 = kate_drive_memory[(kate_drive_memory['brand'] == 'WD') & (kate_drive_memory['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_WD_5:", meanmax_value_WD_5)
print("meanmax_value_WD_5:", meanmin_value_WD_5)


meanmax_value_Segate_5 = kate_drive_memory[(kate_drive_memory['brand'] == 'Segate') & (kate_drive_memory['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Segate_5 = kate_drive_memory[(kate_drive_memory['brand'] == 'Segate') & (kate_drive_memory['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Segate_5:", meanmax_value_Segate_5)
print("meanmax_value_Segate_5:", meanmin_value_Segate_5)

meanmax_value_Corsair_5 = kate_drive_memory[(kate_drive_memory['brand'] == 'Corsair') & (kate_drive_memory['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Corsair_5 = kate_drive_memory[(kate_drive_memory['brand'] == 'Corsair') & (kate_drive_memory['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Corsair_5:", meanmax_value_Corsair_5)
print("meanmax_value_Corsair_5:", meanmin_value_Corsair_5)

meanmax_value_Samsung_5 = kate_drive_memory[(kate_drive_memory['brand'] == 'Samsung') & (kate_drive_memory['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Samsung_5 = kate_drive_memory[(kate_drive_memory['brand'] == 'Samsung') & (kate_drive_memory['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Samsung_5:", meanmax_value_Samsung_5)
print("meanmax_value_Samsung_5:", meanmin_value_Samsung_5)



dmmaxk = kate_drive_memory['prices.amountMax'].mean()
dmmink = kate_drive_memory['prices.amountMin'].mean()

dmstdmax = kate_drive_memory['prices.amountMax'].std()
dmstdmin = kate_drive_memory['prices.amountMin'].std()

meddm = kate_drive_memory['prices.amountMax'].median()

print(dmmaxk)
print(dmmink)

print(dmstdmax)
print(dmstdmin)

# normalization
N_drive_memory = kate_drive_memory.copy()

scaler = MinMaxScaler()
N_drive_memory['prices.amountMax'] = scaler.fit_transform(N_drive_memory[['prices.amountMax']])
N_drive_memory['prices.amountMin'] = scaler.fit_transform(N_drive_memory[['prices.amountMin']])


# remove outlier
Q1_max = N_drive_memory['prices.amountMax'].quantile(0.25)
Q3_max = N_drive_memory['prices.amountMax'].quantile(0.75)
IQR_max = Q3_max - Q1_max
lower_bound_max = Q1_max - 1.5 * IQR_max
upper_bound_max = Q3_max + 1.5 * IQR_max
outliers_max = (N_drive_memory['prices.amountMax'] < lower_bound_max) | (N_drive_memory['prices.amountMax'] > upper_bound_max)


num_outliers_max = outliers_max.sum()
print("Number of outliers:", num_outliers_max)

total_data_points = len(N_drive_memory)
percentage_outliers_max = (num_outliers_max / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_max, "%")


Q1_min = N_drive_memory['prices.amountMin'].quantile(0.25)
Q3_min = N_drive_memory['prices.amountMin'].quantile(0.75)
IQR_min = Q3_min - Q1_min
lower_bound_min = Q1_min - 1.5 * IQR_min
upper_bound_min = Q3_min + 1.5 * IQR_min
outliers_min = (N_drive_memory['prices.amountMin'] < lower_bound_min) | (N_drive_memory['prices.amountMin'] > upper_bound_min)


num_outliers_min = outliers_min.sum()
print("Number of outliers:", num_outliers_min)

total_data_points = len(N_drive_memory)
percentage_outliers_min = (num_outliers_min / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_min, "%")




combined_outliers = outliers_max | outliers_min

N_drive_memory_no_outliers = N_drive_memory[~combined_outliers]

plt.boxplot(N_drive_memory_no_outliers['prices.amountMax'])
plt.show



# check and compare mean values of main competitors in a general scale

meanmax_Corsair_5 = N_drive_memory_no_outliers[(N_drive_memory_no_outliers['brand'] == 'Corsair') & (N_drive_memory_no_outliers['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_Corsair_5 = N_drive_memory_no_outliers[(N_drive_memory_no_outliers['brand'] == 'Corsair') & (N_drive_memory_no_outliers['KVI'] == 5)]['prices.amountMin'].mean()
stdmax_Corsair_5 = N_drive_memory_no_outliers[(N_drive_memory_no_outliers['brand'] == 'Corsair') & (N_drive_memory_no_outliers['KVI'] == 5)]['prices.amountMax'].std()
stdmin_Corsair_5 = N_drive_memory_no_outliers[(N_drive_memory_no_outliers['brand'] == 'Corsair') & (N_drive_memory_no_outliers['KVI'] == 5)]['prices.amountMin'].std()

print("stdmax_Corsair_5:", stdmax_Corsair_5)
print("stdmin_Corsair_5:", stdmin_Corsair_5)
print("meanmax_Corsair_5:", meanmax_Corsair_5)
print("meanmin_Corsair_5:", meanmin_Corsair_5)

meanmax_Samsung_5 = N_drive_memory_no_outliers[(N_drive_memory_no_outliers['brand'] == 'Samsung') & (N_drive_memory_no_outliers['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_Samsung_5 = N_drive_memory_no_outliers[(N_drive_memory_no_outliers['brand'] == 'Samsung') & (N_drive_memory_no_outliers['KVI'] == 5)]['prices.amountMin'].mean()
stdmax_Samsung_5 = N_drive_memory_no_outliers[(N_drive_memory_no_outliers['brand'] == 'Samsung') & (N_drive_memory_no_outliers['KVI'] == 5)]['prices.amountMax'].std()
stdmin_Samsung_5 = N_drive_memory_no_outliers[(N_drive_memory_no_outliers['brand'] == 'Samsung') & (N_drive_memory_no_outliers['KVI'] == 5)]['prices.amountMin'].std()

print("stdmax_Samsung_5:", stdmax_Samsung_5)
print("stdmin_Samsung_5:", stdmin_Samsung_5)
print("meanmax_Samsung_5:", meanmax_Samsung_5)
print("meanmin_Samsung_5:", meanmin_Samsung_5)

dmmaxmean = N_drive_memory_no_outliers['prices.amountMax'].mean()
dmminmean = N_drive_memory_no_outliers['prices.amountMin'].mean()
dmmaxstd = N_drive_memory_no_outliers['prices.amountMax'].std()
dmminstd = N_drive_memory_no_outliers['prices.amountMin'].std()
print("stdmax", dmmaxstd)
print("stdmin", dmminstd)
print("meanmax", dmmaxmean)
print("meanmin", dmminmean)



# TV
kate_TV = merged_df2[merged_df2['CATEGORY'] == 'TV']

plt.boxplot(kate_TV['KVI'])
plt.show

mean_value = kate_TV['KVI'].mean()
std_deviation = kate_TV['KVI'].std()
minimum = kate_TV['KVI'].min()
maximum = kate_TV['KVI'].max()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)
print("Minimum:", minimum)
print("Maximum:", maximum)


meanmax_value_Samsung_5 = kate_TV[(kate_TV['brand'] == 'Samsung') & (kate_TV['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Samsung_5 = kate_TV[(kate_TV['brand'] == 'Samsung') & (kate_TV['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Samsung_5:", meanmax_value_Samsung_5)
print("meanmin_value_Samsung_5:", meanmin_value_Samsung_5)


meanmax_value_Sony_5 = kate_TV[(kate_TV['brand'] == 'Sony') & (kate_TV['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Sony_5 = kate_TV[(kate_TV['brand'] == 'Sony') & (kate_TV['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Sony_5:", meanmax_value_Sony_5)
print("meanmin_value_Sony_5:", meanmin_value_Sony_5)

meanmax_value_LG_5 = kate_TV[(kate_TV['brand'] == 'LG') & (kate_TV['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_LG_5 = kate_TV[(kate_TV['brand'] == 'LG') & (kate_TV['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_LG_5:", meanmax_value_LG_5)
print("meanmin_value_LG_5:", meanmin_value_LG_5)




tvmaxk = kate_TV['prices.amountMax'].mean()
tvmink = kate_TV['prices.amountMin'].mean()

tvstdmax = kate_TV['prices.amountMax'].std()
tvstdmin = kate_TV['prices.amountMin'].std()

medtv = kate_TV['prices.amountMax'].median()

print(tvmaxk)
print(tvmink)

print(tvstdmax)
print(tvstdmin)

# normalization
N_TV = kate_TV.copy()

scaler = MinMaxScaler()
N_TV['prices.amountMax'] = scaler.fit_transform(N_TV[['prices.amountMax']])
N_TV['prices.amountMin'] = scaler.fit_transform(N_TV[['prices.amountMin']])


# remove outlier
Q1_max = N_TV['prices.amountMax'].quantile(0.25)
Q3_max = N_TV['prices.amountMax'].quantile(0.75)
IQR_max = Q3_max - Q1_max
lower_bound_max = Q1_max - 1.5 * IQR_max
upper_bound_max = Q3_max + 1.5 * IQR_max
outliers_max = (N_TV['prices.amountMax'] < lower_bound_max) | (N_TV['prices.amountMax'] > upper_bound_max)


num_outliers_max = outliers_max.sum()
print("Number of outliers:", num_outliers_max)

total_data_points = len(N_TV)
percentage_outliers_max = (num_outliers_max / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_max, "%")


Q1_min = N_TV['prices.amountMin'].quantile(0.25)
Q3_min = N_TV['prices.amountMin'].quantile(0.75)
IQR_min = Q3_min - Q1_min
lower_bound_min = Q1_min - 1.5 * IQR_min
upper_bound_min = Q3_min + 1.5 * IQR_min
outliers_min = (N_TV['prices.amountMin'] < lower_bound_min) | (N_TV['prices.amountMin'] > upper_bound_min)


num_outliers_min = outliers_min.sum()
print("Number of outliers:", num_outliers_min)

total_data_points = len(N_TV)
percentage_outliers_min = (num_outliers_min / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_min, "%")




combined_outliers = outliers_max | outliers_min

N_TV_no_outliers = N_TV[~combined_outliers]

plt.boxplot(N_TV_no_outliers['prices.amountMax'])
plt.show



# check and compare mean values of main competitors in a general scale


meanmax_Samsung_5 = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'Samsung') & (N_TV_no_outliers['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_Samsung_5 = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'Samsung') & (N_TV_no_outliers['KVI'] == 5)]['prices.amountMin'].mean()
stdmax_Samsung_5 = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'Samsung') & (N_TV_no_outliers['KVI'] == 5)]['prices.amountMax'].std()
stdmin_Samsung_5 = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'Samsung') & (N_TV_no_outliers['KVI'] == 5)]['prices.amountMin'].std()

print("stdmax_Samsung_5:", stdmax_Samsung_5)
print("stdmin_Samsung_5:", stdmin_Samsung_5)
print("meanmax_Samsung_5:", meanmax_Samsung_5)
print("meanmin_Samsung_5:", meanmin_Samsung_5)

meanmax_Sony_5 = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'Sony') & (N_TV_no_outliers['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_Sony_5 = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'Sony') & (N_TV_no_outliers['KVI'] == 5)]['prices.amountMin'].mean()
stdmax_Sony_5 = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'Sony') & (N_TV_no_outliers['KVI'] == 5)]['prices.amountMax'].std()
stdmin_Sony_5 = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'Sony') & (N_TV_no_outliers['KVI'] == 5)]['prices.amountMin'].std()

print("stdmax_Sony_5:", stdmax_Sony_5)
print("stdmin_Sony_5:", stdmin_Sony_5)
print("meanmax_Sony_5:", meanmax_Sony_5)
print("meanmin_Sony_5:", meanmin_Sony_5)

meanmax_LG_5 = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'LG') & (N_TV_no_outliers['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_LG_5 = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'LG') & (N_TV_no_outliers['KVI'] == 5)]['prices.amountMin'].mean()
stdmax_LG_5 = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'LG') & (N_TV_no_outliers['KVI'] == 5)]['prices.amountMax'].std()
stdmin_LG_5 = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'LG') & (N_TV_no_outliers['KVI'] == 5)]['prices.amountMin'].std()

print("stdmax_LG_5:", stdmax_LG_5)
print("stdmin_LG_5:", stdmin_LG_5)
print("meanmax_LG_5:", meanmax_LG_5)
print("meanmin_LG_5:", meanmin_LG_5)

tvmaxmean = N_TV_no_outliers['prices.amountMax'].mean()
tvminmean = N_TV_no_outliers['prices.amountMin'].mean()
tvmaxstd = N_TV_no_outliers['prices.amountMax'].std()
tvminstd = N_TV_no_outliers['prices.amountMin'].std()
print("meanmax", tvmaxmean)
print("meanmin", tvminmean)
print("stdmax", tvmaxstd)
print("stdmin", tvminstd)




# Stereo System
kate_StereoSystem = merged_df2[merged_df2['CATEGORY'] == 'StereoSystem']

plt.boxplot(kate_StereoSystem['KVI'])
plt.show

mean_value = kate_StereoSystem['KVI'].mean()
std_deviation = kate_StereoSystem['KVI'].std()
minimum = kate_StereoSystem['KVI'].min()
maximum = kate_StereoSystem['KVI'].max()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)
print("Minimum:", minimum)
print("Maximum:", maximum)


meanmax_value_Yamaha_5 = kate_StereoSystem[(kate_StereoSystem['brand'] == 'Yamaha') & (kate_StereoSystem['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Yamaha_5 = kate_StereoSystem[(kate_StereoSystem['brand'] == 'Yamaha') & (kate_StereoSystem['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Yamaha_5:", meanmax_value_Yamaha_5)
print("meanmin_value_Yamaha_5:", meanmin_value_Yamaha_5)


meanmax_value_Sony_5 = kate_StereoSystem[(kate_StereoSystem['brand'] == 'Sony') & (kate_StereoSystem['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Sony_5 = kate_StereoSystem[(kate_StereoSystem['brand'] == 'Sony') & (kate_StereoSystem['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Sony_5:", meanmax_value_Sony_5)
print("meanmin_value_Sony_5:", meanmin_value_Sony_5)

meanmax_value_Pioneer_5 = kate_StereoSystem[(kate_StereoSystem['brand'] == 'Pioneer') & (kate_StereoSystem['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Pioneer_5 = kate_StereoSystem[(kate_StereoSystem['brand'] == 'Pioneer') & (kate_StereoSystem['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Pioneer_5:", meanmax_value_LG_5)
print("meanmin_value_Pioneer_5:", meanmin_value_LG_5)




StereoSystemmaxk = kate_StereoSystem['prices.amountMax'].mean()
StereoSystemmink = kate_StereoSystem['prices.amountMin'].mean()

StereoSystemstdmax = kate_StereoSystem['prices.amountMax'].std()
StereoSystemstdmin = kate_StereoSystem['prices.amountMin'].std()

medStereoSystem = kate_StereoSystem['prices.amountMax'].median()

print(StereoSystemmaxk)
print(StereoSystemmink)

print(StereoSystemstdmax)
print(StereoSystemstdmin)

# normalization
N_StereoSystem = kate_StereoSystem.copy()

scaler = MinMaxScaler()
N_StereoSystem['prices.amountMax'] = scaler.fit_transform(N_StereoSystem[['prices.amountMax']])
N_StereoSystem['prices.amountMin'] = scaler.fit_transform(N_StereoSystem[['prices.amountMin']])


# remove outlier
Q1_max = N_StereoSystem['prices.amountMax'].quantile(0.25)
Q3_max = N_StereoSystem['prices.amountMax'].quantile(0.75)
IQR_max = Q3_max - Q1_max
lower_bound_max = Q1_max - 1.5 * IQR_max
upper_bound_max = Q3_max + 1.5 * IQR_max
outliers_max = (N_StereoSystem['prices.amountMax'] < lower_bound_max) | (N_StereoSystem['prices.amountMax'] > upper_bound_max)


num_outliers_max = outliers_max.sum()
print("Number of outliers:", num_outliers_max)

total_data_points = len(N_StereoSystem)
percentage_outliers_max = (num_outliers_max / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_max, "%")


Q1_min = N_StereoSystem['prices.amountMin'].quantile(0.25)
Q3_min = N_StereoSystem['prices.amountMin'].quantile(0.75)
IQR_min = Q3_min - Q1_min
lower_bound_min = Q1_min - 1.5 * IQR_min
upper_bound_min = Q3_min + 1.5 * IQR_min
outliers_min = (N_StereoSystem['prices.amountMin'] < lower_bound_min) | (N_StereoSystem['prices.amountMin'] > upper_bound_min)


num_outliers_min = outliers_min.sum()
print("Number of outliers:", num_outliers_min)

total_data_points = len(N_StereoSystem)
percentage_outliers_min = (num_outliers_min / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_min, "%")




combined_outliers = outliers_max | outliers_min

N_StereoSystem_no_outliers = N_StereoSystem[~combined_outliers]

plt.boxplot(N_StereoSystem_no_outliers['prices.amountMax'])
plt.show



# check and compare mean values of main competitors in a general scale




meanmax_Sony_5 = N_StereoSystem_no_outliers[(N_StereoSystem_no_outliers['brand'] == 'Sony') & (N_StereoSystem_no_outliers['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_Sony_5 = N_StereoSystem_no_outliers[(N_StereoSystem_no_outliers['brand'] == 'Sony') & (N_StereoSystem_no_outliers['KVI'] == 5)]['prices.amountMin'].mean()
stdmax_Sony_5 = N_StereoSystem_no_outliers[(N_StereoSystem_no_outliers['brand'] == 'Sony') & (N_StereoSystem_no_outliers['KVI'] == 5)]['prices.amountMax'].std()
stdmin_Sony_5 = N_StereoSystem_no_outliers[(N_StereoSystem_no_outliers['brand'] == 'Sony') & (N_StereoSystem_no_outliers['KVI'] == 5)]['prices.amountMin'].std()
print("stdmax_Sony_5:", stdmax_Sony_5)
print("stdmin_Sony_5:", stdmin_Sony_5)
print("meanmax_Sony_5:", meanmax_Sony_5)
print("meanmin_Sony_5:", meanmin_Sony_5)

meanmax_Pioneer_5 = N_StereoSystem_no_outliers[(N_StereoSystem_no_outliers['brand'] == 'Pioneer') & (N_StereoSystem_no_outliers['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_Pioneer_5 = N_StereoSystem_no_outliers[(N_StereoSystem_no_outliers['brand'] == 'Pioneer') & (N_StereoSystem_no_outliers['KVI'] == 5)]['prices.amountMin'].mean()
stdmax_Pioneer_5 = N_StereoSystem_no_outliers[(N_StereoSystem_no_outliers['brand'] == 'Pioneer') & (N_StereoSystem_no_outliers['KVI'] == 5)]['prices.amountMax'].std()
stdmin_Pioneer_5 = N_StereoSystem_no_outliers[(N_StereoSystem_no_outliers['brand'] == 'Pioneer') & (N_StereoSystem_no_outliers['KVI'] == 5)]['prices.amountMin'].std()
print("stdmax_Pioneer_5:", stdmax_Pioneer_5)
print("stdmin_Pioneer_5:", stdmin_Pioneer_5)
print("meanmax_Pioneer_5:", meanmax_Pioneer_5)
print("meanmin_Pioneer_5:", meanmin_Pioneer_5)

StereoSystemmaxmean = N_StereoSystem_no_outliers['prices.amountMax'].mean()
StereoSystemminmean = N_StereoSystem_no_outliers['prices.amountMin'].mean()
StereoSystemmaxstd = N_StereoSystem_no_outliers['prices.amountMax'].std()
StereoSystemminstd = N_StereoSystem_no_outliers['prices.amountMin'].std()
print("meanmax", StereoSystemmaxmean)
print("meanmin", StereoSystemminmean)
print("stdmax", StereoSystemmaxstd)
print("stdmin", StereoSystemminstd)




# Camera
kate_Camera = merged_df2[merged_df2['CATEGORY'] == 'Camera']

plt.boxplot(kate_Camera['KVI'])
plt.show

mean_value = kate_Camera['KVI'].mean()
std_deviation = kate_Camera['KVI'].std()
minimum = kate_Camera['KVI'].min()
maximum = kate_Camera['KVI'].max()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)
print("Minimum:", minimum)
print("Maximum:", maximum)



meanmax_value_Sony_5 = kate_Camera[(kate_Camera['brand'] == 'Sony') & (kate_Camera['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Sony_5 = kate_Camera[(kate_Camera['brand'] == 'Sony') & (kate_Camera['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Sony_5:", meanmax_value_Sony_5)
print("meanmin_value_Sony_5:", meanmin_value_Sony_5)

meanmax_value_Canon_5 = kate_Camera[(kate_Camera['brand'] == 'Canon') & (kate_Camera['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Canon_5 = kate_Camera[(kate_Camera['brand'] == 'Canon') & (kate_Camera['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Canon_5:", meanmax_value_Canon_5)
print("meanmin_value_Canon_5:", meanmin_value_Canon_5)

meanmax_value_Nikon_5 = kate_Camera[(kate_Camera['brand'] == 'Nikon') & (kate_Camera['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Nikon_5 = kate_Camera[(kate_Camera['brand'] == 'Nikon') & (kate_Camera['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Nikon_5:", meanmax_value_Nikon_5)
print("meanmin_value_Nikon_5:", meanmin_value_Nikon_5)


meanmax_value_Fujifilm_5 = kate_Camera[(kate_Camera['brand'] == 'Fujifilm') & (kate_Camera['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Fujifilm_5 = kate_Camera[(kate_Camera['brand'] == 'Fujifilm') & (kate_Camera['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Fujifilm_5:", meanmax_value_Fujifilm_5)
print("meanmin_value_Fujifilm_5:", meanmin_value_Fujifilm_5)




Cameramaxk = kate_Camera['prices.amountMax'].mean()
Cameramink = kate_Camera['prices.amountMin'].mean()

Camerastdmax = kate_Camera['prices.amountMax'].std()
Camerastdmin = kate_Camera['prices.amountMin'].std()

medCamera = kate_Camera['prices.amountMax'].median()

print(Cameramaxk)
print(Cameramink)

print(Camerastdmax)
print(Camerastdmin)

# normalization
N_Camera = kate_Camera.copy()

scaler = MinMaxScaler()
N_Camera['prices.amountMax'] = scaler.fit_transform(N_Camera[['prices.amountMax']])
N_Camera['prices.amountMin'] = scaler.fit_transform(N_Camera[['prices.amountMin']])


# remove outlier
Q1_max = N_Camera['prices.amountMax'].quantile(0.25)
Q3_max = N_Camera['prices.amountMax'].quantile(0.75)
IQR_max = Q3_max - Q1_max
lower_bound_max = Q1_max - 1.5 * IQR_max
upper_bound_max = Q3_max + 1.5 * IQR_max
outliers_max = (N_Camera['prices.amountMax'] < lower_bound_max) | (N_Camera['prices.amountMax'] > upper_bound_max)


num_outliers_max = outliers_max.sum()
print("Number of outliers:", num_outliers_max)

total_data_points = len(N_Camera)
percentage_outliers_max = (num_outliers_max / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_max, "%")


Q1_min = N_Camera['prices.amountMin'].quantile(0.25)
Q3_min = N_Camera['prices.amountMin'].quantile(0.75)
IQR_min = Q3_min - Q1_min
lower_bound_min = Q1_min - 1.5 * IQR_min
upper_bound_min = Q3_min + 1.5 * IQR_min
outliers_min = (N_Camera['prices.amountMin'] < lower_bound_min) | (N_Camera['prices.amountMin'] > upper_bound_min)


num_outliers_min = outliers_min.sum()
print("Number of outliers:", num_outliers_min)

total_data_points = len(N_Camera)
percentage_outliers_min = (num_outliers_min / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_min, "%")




combined_outliers = outliers_max | outliers_min

N_Camera_no_outliers = N_Camera[~combined_outliers]

plt.boxplot(N_Camera_no_outliers['prices.amountMax'])
plt.show



# check and compare mean values of main competitors in a general scale



meanmax_Sony_5 = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Sony') & (N_Camera_no_outliers['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_Sony_5 = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Sony') & (N_Camera_no_outliers['KVI'] == 5)]['prices.amountMin'].mean()
stdmax_Sony_5 = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Sony') & (N_Camera_no_outliers['KVI'] == 5)]['prices.amountMax'].std()
stdmin_Sony_5 = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Sony') & (N_Camera_no_outliers['KVI'] == 5)]['prices.amountMin'].std()
print("stdmax_Sony_5:", stdmax_Sony_5)
print("stdmin_Sony_5:", stdmin_Sony_5)
print("meanmax_Sony_5:", meanmax_Sony_5)
print("meanmin_Sony_5:", meanmin_Sony_5)

meanmax_Canon_5 = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Canon') & (N_Camera_no_outliers['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_Canon_5 = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Canon') & (N_Camera_no_outliers['KVI'] == 5)]['prices.amountMin'].mean()
stdmax_Canon_5 = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Canon') & (N_Camera_no_outliers['KVI'] == 5)]['prices.amountMax'].std()
stdmin_Canon_5 = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Canon') & (N_Camera_no_outliers['KVI'] == 5)]['prices.amountMin'].std()
print("stdmax_Canon_5:", stdmax_Canon_5)
print("stdmin_Canon_5:", stdmin_Canon_5)
print("meanmax_Canon_5:", meanmax_Canon_5)
print("meanmin_Canon_5:", meanmin_Canon_5)

Cameramaxmean = N_Camera_no_outliers['prices.amountMax'].mean()
Cameraminmean = N_Camera_no_outliers['prices.amountMin'].mean()
Cameramaxstd = N_Camera_no_outliers['prices.amountMax'].std()
Cameraminstd = N_Camera_no_outliers['prices.amountMin'].std()
print("meanmax", Cameramaxmean)
print("meanmin", Cameraminmean)
print("stdmax", Cameramaxstd)
print("stdmin", Cameraminstd)





# laptop
kate_Laptop = merged_df2[merged_df2['CATEGORY'] == 'Laptop']

plt.boxplot(kate_Laptop['KVI'])
plt.show

mean_value = kate_Laptop['KVI'].mean()
std_deviation = kate_Laptop['KVI'].std()
minimum = kate_Laptop['KVI'].min()
maximum = kate_Laptop['KVI'].max()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)
print("Minimum:", minimum)
print("Maximum:", maximum)


meanmax_value_Apple_5 = kate_Laptop[(kate_Laptop['brand'] == 'Apple') & (kate_Laptop['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Apple_5 = kate_Laptop[(kate_Laptop['brand'] == 'Apple') & (kate_Laptop['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Apple_5:", meanmax_value_Apple_5)
print("meanmin_value_Apple_5:", meanmin_value_Apple_5)


meanmax_value_Asus_5 = kate_Laptop[(kate_Laptop['brand'] == 'Asus') & (kate_Laptop['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Asus_5 = kate_Laptop[(kate_Laptop['brand'] == 'Asus') & (kate_Laptop['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Asus_5:", meanmax_value_Asus_5)
print("meanmin_value_Asus_5:", meanmin_value_Asus_5)



Laptopmaxk = kate_Laptop['prices.amountMax'].mean()
Laptopmink = kate_Laptop['prices.amountMin'].mean()

Laptopstdmax = kate_Laptop['prices.amountMax'].std()
Laptopstdmin = kate_Laptop['prices.amountMin'].std()

medLaptop = kate_Laptop['prices.amountMax'].median()

print(Laptopmaxk)
print(Laptopmink)

print(Laptopstdmax)
print(Laptopstdmin)

# normalization
N_Laptop = kate_Laptop.copy()

scaler = MinMaxScaler()
N_Laptop['prices.amountMax'] = scaler.fit_transform(N_Laptop[['prices.amountMax']])
N_Laptop['prices.amountMin'] = scaler.fit_transform(N_Laptop[['prices.amountMin']])


# remove outlier
Q1_max = N_Laptop['prices.amountMax'].quantile(0.25)
Q3_max = N_Laptop['prices.amountMax'].quantile(0.75)
IQR_max = Q3_max - Q1_max
lower_bound_max = Q1_max - 1.5 * IQR_max
upper_bound_max = Q3_max + 1.5 * IQR_max
outliers_max = (N_Laptop['prices.amountMax'] < lower_bound_max) | (N_Laptop['prices.amountMax'] > upper_bound_max)


num_outliers_max = outliers_max.sum()
print("Number of outliers:", num_outliers_max)

total_data_points = len(N_Laptop)
percentage_outliers_max = (num_outliers_max / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_max, "%")


Q1_min = N_Laptop['prices.amountMin'].quantile(0.25)
Q3_min = N_Laptop['prices.amountMin'].quantile(0.75)
IQR_min = Q3_min - Q1_min
lower_bound_min = Q1_min - 1.5 * IQR_min
upper_bound_min = Q3_min + 1.5 * IQR_min
outliers_min = (N_Laptop['prices.amountMin'] < lower_bound_min) | (N_Laptop['prices.amountMin'] > upper_bound_min)


num_outliers_min = outliers_min.sum()
print("Number of outliers:", num_outliers_min)

total_data_points = len(N_Laptop)
percentage_outliers_min = (num_outliers_min / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_min, "%")




combined_outliers = outliers_max | outliers_min

N_Laptop_no_outliers = N_Laptop[~combined_outliers]

plt.boxplot(N_Laptop_no_outliers['prices.amountMax'])
plt.show



# check and compare mean values of main competitors in a general scale


meanmax_Asus_5 = N_Laptop_no_outliers[(N_Laptop_no_outliers['brand'] == 'Asus') & (N_Laptop_no_outliers['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_Asus_5 = N_Laptop_no_outliers[(N_Laptop_no_outliers['brand'] == 'Asus') & (N_Laptop_no_outliers['KVI'] == 5)]['prices.amountMin'].mean()
stdmax_Asus_5 = N_Laptop_no_outliers[(N_Laptop_no_outliers['brand'] == 'Asus') & (N_Laptop_no_outliers['KVI'] == 5)]['prices.amountMax'].std()
stdmin_Asus_5 = N_Laptop_no_outliers[(N_Laptop_no_outliers['brand'] == 'Asus') & (N_Laptop_no_outliers['KVI'] == 5)]['prices.amountMin'].std()
print("stdmax_Asus_5:", stdmax_Asus_5)
print("stdmin_Asus_5:", stdmin_Asus_5)
print("meanmax_Asus_5:", meanmax_Asus_5)
print("meanmin_Asus_5:", meanmin_Asus_5)


Laptopmaxmean = N_Laptop_no_outliers['prices.amountMax'].mean()
Laptopminmean = N_Laptop_no_outliers['prices.amountMin'].mean()
Laptopmaxstd = N_Laptop_no_outliers['prices.amountMax'].std()
Laptopminstd = N_Laptop_no_outliers['prices.amountMin'].std()

print("meanmax", Laptopmaxmean)
print("meanmin", Laptopminmean)
print("stdmax", Laptopmaxstd)
print("stdmin", Laptopminstd)



# Keyboard& mouse

kate_Keyboard_Mouse = merged_df2[merged_df2['CATEGORY'] == 'Keyboard_Mouse']

plt.boxplot(kate_Keyboard_Mouse['KVI'])
plt.show

mean_value = kate_Keyboard_Mouse['KVI'].mean()
std_deviation = kate_Keyboard_Mouse['KVI'].std()
minimum = kate_Keyboard_Mouse['KVI'].min()
maximum = kate_Keyboard_Mouse['KVI'].max()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)
print("Minimum:", minimum)
print("Maximum:", maximum)


meanmax_value_Logitech_5 = kate_Keyboard_Mouse[(kate_Keyboard_Mouse['brand'] == 'Logitech') & (kate_Keyboard_Mouse['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Logitech_5 = kate_Keyboard_Mouse[(kate_Keyboard_Mouse['brand'] == 'Logitech') & (kate_Keyboard_Mouse['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Logitech_5:", meanmax_value_Logitech_5)
print("meanmin_value_Logitech_5:", meanmin_value_Logitech_5)


meanmax_value_Corsair_5 = kate_Keyboard_Mouse[(kate_Keyboard_Mouse['brand'] == 'Corsair') & (kate_Keyboard_Mouse['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Corsair_5 = kate_Keyboard_Mouse[(kate_Keyboard_Mouse['brand'] == 'Corsair') & (kate_Keyboard_Mouse['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Corsair_5:", meanmax_value_Corsair_5)
print("meanmin_value_Corsair_5:", meanmin_value_Corsair_5)

meanmax_value_Razer_5 = kate_Keyboard_Mouse[(kate_Keyboard_Mouse['brand'] == 'Razer') & (kate_Keyboard_Mouse['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Razer_5 = kate_Keyboard_Mouse[(kate_Keyboard_Mouse['brand'] == 'Razer') & (kate_Keyboard_Mouse['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Razer_5:", meanmax_value_Razer_5)
print("meanmin_value_Razer_5:", meanmin_value_Razer_5)

meanmax_value_Microsoft_5 = kate_Keyboard_Mouse[(kate_Keyboard_Mouse['brand'] == 'Microsoft') & (kate_Keyboard_Mouse['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_value_Microsoft_5 = kate_Keyboard_Mouse[(kate_Keyboard_Mouse['brand'] == 'Microsoft') & (kate_Keyboard_Mouse['KVI'] == 5)]['prices.amountMin'].mean()

print("meanmax_value_Microsoft_5:", meanmax_value_Microsoft_5)
print("meanmin_value_Microsoft_5:", meanmin_value_Microsoft_5)



Keyboard_Mousemaxk = kate_Keyboard_Mouse['prices.amountMax'].mean()
Keyboard_Mousemink = kate_Keyboard_Mouse['prices.amountMin'].mean()

Keyboard_Mousestdmax = kate_Keyboard_Mouse['prices.amountMax'].std()
Keyboard_Mousestdmin = kate_Keyboard_Mouse['prices.amountMin'].std()

medKeyboard_Mouse = kate_Keyboard_Mouse['prices.amountMax'].median()

print(Keyboard_Mousemaxk)
print(Keyboard_Mousemink)

print(Keyboard_Mousestdmax)
print(Keyboard_Mousestdmin)

# normalization
N_Keyboard_Mouse = kate_Keyboard_Mouse.copy()

scaler = MinMaxScaler()
N_Keyboard_Mouse['prices.amountMax'] = scaler.fit_transform(N_Keyboard_Mouse[['prices.amountMax']])
N_Keyboard_Mouse['prices.amountMin'] = scaler.fit_transform(N_Keyboard_Mouse[['prices.amountMin']])


# remove outlier
Q1_max = N_Keyboard_Mouse['prices.amountMax'].quantile(0.25)
Q3_max = N_Keyboard_Mouse['prices.amountMax'].quantile(0.75)
IQR_max = Q3_max - Q1_max
lower_bound_max = Q1_max - 1.5 * IQR_max
upper_bound_max = Q3_max + 1.5 * IQR_max
outliers_max = (N_Keyboard_Mouse['prices.amountMax'] < lower_bound_max) | (N_Keyboard_Mouse['prices.amountMax'] > upper_bound_max)


num_outliers_max = outliers_max.sum()
print("Number of outliers:", num_outliers_max)

total_data_points = len(N_Keyboard_Mouse)
percentage_outliers_max = (num_outliers_max / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_max, "%")


Q1_min = N_Keyboard_Mouse['prices.amountMin'].quantile(0.25)
Q3_min = N_Keyboard_Mouse['prices.amountMin'].quantile(0.75)
IQR_min = Q3_min - Q1_min
lower_bound_min = Q1_min - 1.5 * IQR_min
upper_bound_min = Q3_min + 1.5 * IQR_min
outliers_min = (N_Keyboard_Mouse['prices.amountMin'] < lower_bound_min) | (N_Keyboard_Mouse['prices.amountMin'] > upper_bound_min)


num_outliers_min = outliers_min.sum()
print("Number of outliers:", num_outliers_min)

total_data_points = len(N_Keyboard_Mouse)
percentage_outliers_min = (num_outliers_min / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_min, "%")




combined_outliers = outliers_max | outliers_min

N_Keyboard_Mouse_no_outliers = N_Keyboard_Mouse[~combined_outliers]

plt.boxplot(N_Keyboard_Mouse_no_outliers['prices.amountMax'])
plt.show



# check and compare mean values of main competitors in a general scale


meanmax_Logitech_5 = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Logitech') & (N_Keyboard_Mouse_no_outliers['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_Logitech_5 = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Logitech') & (N_Keyboard_Mouse_no_outliers['KVI'] == 5)]['prices.amountMin'].mean()
stdmax_Logitech_5 = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Logitech') & (N_Keyboard_Mouse_no_outliers['KVI'] == 5)]['prices.amountMax'].std()
stdmin_Logitech_5 = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Logitech') & (N_Keyboard_Mouse_no_outliers['KVI'] == 5)]['prices.amountMin'].std()
print("stdmax_Logitech_5:", stdmax_Logitech_5)
print("stdmin_Logitech_5:", stdmin_Logitech_5)
print("meanmax_Logitech_5:", meanmax_Logitech_5)
print("meanmin_Logitech_5:", meanmin_Logitech_5)

meanmax_Corsair_5 = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Corsair') & (N_Keyboard_Mouse_no_outliers['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_Corsair_5 = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Corsair') & (N_Keyboard_Mouse_no_outliers['KVI'] == 5)]['prices.amountMin'].mean()
stdmax_Corsair_5 = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Corsair') & (N_Keyboard_Mouse_no_outliers['KVI'] == 5)]['prices.amountMax'].std()
stdmin_Corsair_5 = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Corsair') & (N_Keyboard_Mouse_no_outliers['KVI'] == 5)]['prices.amountMin'].std()
print("stdmax_Corsair_5:", stdmax_Corsair_5)
print("stdmin_Corsair_5:", stdmin_Corsair_5)
print("meanmax_Corsair_5:", meanmax_Corsair_5)
print("meanmin_Corsair_5:", meanmin_Corsair_5)

meanmax_Razer_5 = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Razer') & (N_Keyboard_Mouse_no_outliers['KVI'] == 5)]['prices.amountMax'].mean()
meanmin_Razer_5 = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Razer') & (N_Keyboard_Mouse_no_outliers['KVI'] == 5)]['prices.amountMin'].mean()
stdmax_Razer_5 = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Razer') & (N_Keyboard_Mouse_no_outliers['KVI'] == 5)]['prices.amountMax'].std()
stdmin_Razer_5 = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Razer') & (N_Keyboard_Mouse_no_outliers['KVI'] == 5)]['prices.amountMin'].std()
print("stdmax_Razer_5:", stdmax_Razer_5)
print("stdmin_Razer_5:", stdmin_Razer_5)
print("meanmax_Razer_5:", meanmax_Razer_5)
print("meanmin_Razer_5:", meanmin_Razer_5)

Keyboard_Mousemaxmean = N_Keyboard_Mouse_no_outliers['prices.amountMax'].mean()
Keyboard_Mouseminmean = N_Keyboard_Mouse_no_outliers['prices.amountMin'].mean()
Keyboard_Mousemaxstd = N_Keyboard_Mouse_no_outliers['prices.amountMax'].std()
Keyboard_Mouseminstd = N_Keyboard_Mouse_no_outliers['prices.amountMin'].std()
print("meanmax", Keyboard_Mousemaxmean)
print("meanmin", Keyboard_Mouseminmean)
print("stdmax", Keyboard_Mousemaxstd)
print("stdmin", Keyboard_Mouseminstd)



### KVI model 2 -> feature importance

merged_df2['KVI2'] = 0

merged_df2.info()


X = merged_df2.drop(['prices.condition','id','prices.availability', 'prices.amountMax','prices.amountMin','prices.currency','prices.dateSeen','asins','categories','dateAdded','dateUpdated','keys','name','SdateUpdated','SdateAdded','KVI','prices.isSale','KVI2'], axis=1)

y = merged_df2['prices.isSale']

X_encoded = pd.get_dummies(X, columns=['prices.merchant','brand','prices.shipping','CATEGORY'])


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

feature_importance = clf.feature_importances_


importance_df = pd.DataFrame({'prices.isSale': X_encoded.columns, 'Importance': feature_importance})

# Sort the DataFrame by importance values
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df.head(15))

# Create a bar plot

plt.figure(figsize=(10, 6))
plt.bar(importance_df['prices.isSale'], importance_df['Importance'])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance from Decision Tree Classifier')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Scaled Im-portance

data = [0.391607, 0.071238, 0.052709, 0.050020, 0.038062, 0.036212, 0.024984, 0.015161, 0.014983, 0.014290, 0.012715, 0.012239]

min_val = min(data)

scaled_data = [1 + (x - min_val) for x in data]



grouped_data2 = merged_df2.groupby(by='CATEGORY')
MCM_mean = grouped_data2['MCM'].mean()
MCM_mean.mean()


merged_df2['KVI2'] = 0.000

merged_df2.info()



for index, row in merged_df2.iterrows():
    score2 = 0
    
    if (row['diff'] > 0) or (row['diff'] >= row['Mean diff']):
        score2 += 1.379
    if row['Merch Count'] >= row['MCM']:
        score2 += 1.059
    if row['Mean diff'] >= row['diffC']:
        score2 += 1.040
    if row['prices.merchant'] == 'Bestbuy.com':   
        score2 += 1.038
    if row['MCM'] >= 5.034623536613865:
        score2 += 1.026
    if row['prices.merchant'] == 'bhphotovideo.com':   
        score2 += 1.024   
    if row['prices.shipping'] == 'FREE':   
        score2 += 1.013
    if row['prices.shipping'] == 'Value':   
        score2 += 1.003  
    if row['prices.merchant'] == 'Amazon.com':   
        score2 += 1.003   
    if row['brand'] == 'Sony':   
        score2 += 1.002
    if row['prices.merchant'] == 'Walmart.com':   
        score2 += 1   
    if row['prices.shipping'] == 'Free Expedited Shipping':   
        score2 += 1
    
    merged_df2.at[index, 'KVI2'] = row['KVI2'] + score2



## KVI score check in general scale
plt.boxplot(merged_df2['KVI2'])
plt.show

mean_value = merged_df2['KVI2'].mean()
std_deviation = merged_df2['KVI2'].std()
minimum = merged_df2['KVI2'].min()
maximum = merged_df2['KVI2'].max()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)
print("Minimum:", minimum)
print("Maximum:", maximum)

merged_df2.to_csv('D:/BA_Dissertation/script/merged_df22.csv', index=False)



merged_df2

merged_df2['dkvi'] = merged_df2['KVI2'] - merged_df2['KVI']


plt.boxplot(merged_df2['dkvi'])
plt.show

merged_df2['dkvi']



_Headphone_Speaker = merged_df2[merged_df2['CATEGORY'] == 'Headphone_Speaker']
_Drive_Memory = merged_df2[merged_df2['CATEGORY'] == 'Drive_Memory']
_TV = merged_df2[merged_df2['CATEGORY'] == 'TV']
_StereoSystem = merged_df2[merged_df2['CATEGORY'] == 'StereoSystem']
_Camera = merged_df2[merged_df2['CATEGORY'] == 'Camera']
_Laptop = merged_df2[merged_df2['CATEGORY'] == 'Laptop']
_Keyboard_Mouse = merged_df2[merged_df2['CATEGORY'] == 'Keyboard_Mouse']






plt.boxplot(_Headphone_Speaker['dkvi'])
plt.show

plt.boxplot(_Drive_Memory['dkvi'])
plt.show

plt.boxplot(_TV['dkvi'])
plt.show

plt.boxplot(_StereoSystem['dkvi'])
plt.show

plt.boxplot(_Camera['dkvi'])
plt.show

plt.boxplot(_Laptop['dkvi'])
plt.show

plt.boxplot(_Keyboard_Mouse['dkvi'])
plt.show



# headphone& speaker

mean_value = _Headphone_Speaker['KVI2'].mean()
std_deviation = _Headphone_Speaker['KVI2'].std()
minimum = _Headphone_Speaker['KVI2'].min()
maximum = _Headphone_Speaker['KVI2'].max()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)
print("Minimum:", minimum)
print("Maximum:", maximum)

# normalization
N_Headphone_Speaker = _Headphone_Speaker.copy()

scaler = MinMaxScaler()
N_Headphone_Speaker['prices.amountMax'] = scaler.fit_transform(N_Headphone_Speaker[['prices.amountMax']])
N_Headphone_Speaker['prices.amountMin'] = scaler.fit_transform(N_Headphone_Speaker[['prices.amountMin']])


# remove outlier
Q1_max = N_Headphone_Speaker['prices.amountMax'].quantile(0.25)
Q3_max = N_Headphone_Speaker['prices.amountMax'].quantile(0.75)
IQR_max = Q3_max - Q1_max
lower_bound_max = Q1_max - 1.5 * IQR_max
upper_bound_max = Q3_max + 1.5 * IQR_max
outliers_max = (N_Headphone_Speaker['prices.amountMax'] < lower_bound_max) | (N_Headphone_Speaker['prices.amountMax'] > upper_bound_max)

num_outliers_max = outliers_max.sum()
print("Number of outliers:", num_outliers_max)

total_data_points = len(N_Headphone_Speaker)
percentage_outliers_max = (num_outliers_max / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_max, "%")


Q1_min = N_Headphone_Speaker['prices.amountMin'].quantile(0.25)
Q3_min = N_Headphone_Speaker['prices.amountMin'].quantile(0.75)
IQR_min = Q3_min - Q1_min
lower_bound_min = Q1_min - 1.5 * IQR_min
upper_bound_min = Q3_min + 1.5 * IQR_min
outliers_min = (N_Headphone_Speaker['prices.amountMin'] < lower_bound_min) | (N_Headphone_Speaker['prices.amountMin'] > upper_bound_min)

num_outliers_min = outliers_min.sum()
print("Number of outliers:", num_outliers_min)

total_data_points = len(N_Headphone_Speaker)
percentage_outliers_min = (num_outliers_min / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_min, "%")


combined_outliers = outliers_max | outliers_min

N_Headphone_Speaker_no_outliers = N_Headphone_Speaker[~combined_outliers]

plt.boxplot(N_Headphone_Speaker_no_outliers['prices.amountMax'])
plt.show


# check and compare mean values of main competitors in a general scale

meanmax_Sony_ = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'Sony') & (N_Headphone_Speaker_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_Sony_ = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'Sony') & (N_Headphone_Speaker_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_Sony_ = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'Sony') & (N_Headphone_Speaker_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_Sony_ = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'Sony') & (N_Headphone_Speaker_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_Sony_:", stdmax_Sony_)
print("stdmin_Sony_:", stdmin_Sony_)
print("meanmax_Sony_:", meanmax_Sony_)
print("meanmin_Sony_:", meanmin_Sony_)


meanmax_JBL_ = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'JBL') & (N_Headphone_Speaker_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_JBL_ = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'JBL') & (N_Headphone_Speaker_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_JBL_ = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'JBL') & (N_Headphone_Speaker_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_JBL_ = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'JBL') & (N_Headphone_Speaker_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_JBL_:", stdmax_JBL_)
print("stdmin_JBL_:", stdmin_JBL_)
print("meanmax_JBL_:", meanmax_JBL_)
print("meanmax_JBL_:", meanmin_JBL_)


meanmax_JBL_ = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'JBL') & (N_Headphone_Speaker_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_JBL_ = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'JBL') & (N_Headphone_Speaker_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_JBL_ = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'JBL') & (N_Headphone_Speaker_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_JBL_ = N_Headphone_Speaker_no_outliers[(N_Headphone_Speaker_no_outliers['brand'] == 'JBL') & (N_Headphone_Speaker_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_JBL_:", stdmax_JBL_)
print("stdmin_JBL_:", stdmin_JBL_)
print("meanmax_JBL_:", meanmax_JBL_)
print("meanmax_JBL_:", meanmin_JBL_)


# stereo system

mean_value = _StereoSystem['KVI2'].mean()
std_deviation = _StereoSystem['KVI2'].std()
minimum = _StereoSystem['KVI2'].min()
maximum = _StereoSystem['KVI2'].max()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)
print("Minimum:", minimum)
print("Maximum:", maximum)

# normalization
N_StereoSystem = _StereoSystem.copy()

scaler = MinMaxScaler()
N_StereoSystem['prices.amountMax'] = scaler.fit_transform(N_StereoSystem[['prices.amountMax']])
N_StereoSystem['prices.amountMin'] = scaler.fit_transform(N_StereoSystem[['prices.amountMin']])


# remove outlier
Q1_max = N_StereoSystem['prices.amountMax'].quantile(0.25)
Q3_max = N_StereoSystem['prices.amountMax'].quantile(0.75)
IQR_max = Q3_max - Q1_max
lower_bound_max = Q1_max - 1.5 * IQR_max
upper_bound_max = Q3_max + 1.5 * IQR_max
outliers_max = (N_StereoSystem['prices.amountMax'] < lower_bound_max) | (N_StereoSystem['prices.amountMax'] > upper_bound_max)

num_outliers_max = outliers_max.sum()
print("Number of outliers:", num_outliers_max)

total_data_points = len(N_StereoSystem)
percentage_outliers_max = (num_outliers_max / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_max, "%")


Q1_min = N_StereoSystem['prices.amountMin'].quantile(0.25)
Q3_min = N_StereoSystem['prices.amountMin'].quantile(0.75)
IQR_min = Q3_min - Q1_min
lower_bound_min = Q1_min - 1.5 * IQR_min
upper_bound_min = Q3_min + 1.5 * IQR_min
outliers_min = (N_StereoSystem['prices.amountMin'] < lower_bound_min) | (N_StereoSystem['prices.amountMin'] > upper_bound_min)

num_outliers_min = outliers_min.sum()
print("Number of outliers:", num_outliers_min)

total_data_points = len(N_StereoSystem)
percentage_outliers_min = (num_outliers_min / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_min, "%")


combined_outliers = outliers_max | outliers_min

N_StereoSystem_no_outliers = N_StereoSystem[~combined_outliers]

plt.boxplot(N_StereoSystem_no_outliers['prices.amountMax'])
plt.show


# check and compare mean values of main competitors in a general scale

meanmax_Sony_ = N_StereoSystem_no_outliers[(N_StereoSystem_no_outliers['brand'] == 'Sony') & (N_StereoSystem_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_Sony_ = N_StereoSystem_no_outliers[(N_StereoSystem_no_outliers['brand'] == 'Sony') & (N_StereoSystem_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_Sony_ = N_StereoSystem_no_outliers[(N_StereoSystem_no_outliers['brand'] == 'Sony') & (N_StereoSystem_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_Sony_ = N_StereoSystem_no_outliers[(N_StereoSystem_no_outliers['brand'] == 'Sony') & (N_StereoSystem_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_Sony_:", stdmax_Sony_)
print("stdmin_Sony_:", stdmin_Sony_)
print("meanmax_Sony_:", meanmax_Sony_)
print("meanmin_Sony_:", meanmin_Sony_)


# TV

mean_value = _TV['KVI2'].mean()
std_deviation = _TV['KVI2'].std()
minimum = _TV['KVI2'].min()
maximum = _TV['KVI2'].max()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)
print("Minimum:", minimum)
print("Maximum:", maximum)

# normalization
N_TV = _TV.copy()

scaler = MinMaxScaler()
N_TV['prices.amountMax'] = scaler.fit_transform(N_TV[['prices.amountMax']])
N_TV['prices.amountMin'] = scaler.fit_transform(N_TV[['prices.amountMin']])


# remove outlier
Q1_max = N_TV['prices.amountMax'].quantile(0.25)
Q3_max = N_TV['prices.amountMax'].quantile(0.75)
IQR_max = Q3_max - Q1_max
lower_bound_max = Q1_max - 1.5 * IQR_max
upper_bound_max = Q3_max + 1.5 * IQR_max
outliers_max = (N_TV['prices.amountMax'] < lower_bound_max) | (N_TV['prices.amountMax'] > upper_bound_max)

num_outliers_max = outliers_max.sum()
print("Number of outliers:", num_outliers_max)

total_data_points = len(N_TV)
percentage_outliers_max = (num_outliers_max / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_max, "%")


Q1_min = N_TV['prices.amountMin'].quantile(0.25)
Q3_min = N_TV['prices.amountMin'].quantile(0.75)
IQR_min = Q3_min - Q1_min
lower_bound_min = Q1_min - 1.5 * IQR_min
upper_bound_min = Q3_min + 1.5 * IQR_min
outliers_min = (N_TV['prices.amountMin'] < lower_bound_min) | (N_TV['prices.amountMin'] > upper_bound_min)

num_outliers_min = outliers_min.sum()
print("Number of outliers:", num_outliers_min)

total_data_points = len(N_TV)
percentage_outliers_min = (num_outliers_min / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_min, "%")


combined_outliers = outliers_max | outliers_min

N_TV_no_outliers = N_TV[~combined_outliers]

plt.boxplot(N_TV_no_outliers['prices.amountMax'])
plt.show


# check and compare mean values of main competitors in a general scale

meanmax_Sony_ = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'Sony') & (N_TV_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_Sony_ = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'Sony') & (N_TV_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_Sony_ = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'Sony') & (N_TV_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_Sony_ = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'Sony') & (N_TV_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_Sony_:", stdmax_Sony_)
print("stdmin_Sony_:", stdmin_Sony_)
print("meanmax_Sony_:", meanmax_Sony_)
print("meanmin_Sony_:", meanmin_Sony_)

meanmax_Samsung_ = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'Samsung') & (N_TV_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_Samsung_ = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'Samsung') & (N_TV_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_Samsung_ = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'Samsung') & (N_TV_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_Samsung_ = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'Samsung') & (N_TV_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_Samsung_:", stdmax_Samsung_)
print("stdmin_Samsung_:", stdmin_Samsung_)
print("meanmax_Samsung_:", meanmax_Samsung_)
print("meanmin_Samsung_:", meanmin_Samsung_)

meanmax_LG_ = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'LG') & (N_TV_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_LG_ = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'LG') & (N_TV_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_LG_ = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'LG') & (N_TV_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_LG_ = N_TV_no_outliers[(N_TV_no_outliers['brand'] == 'LG') & (N_TV_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_LG_:", stdmax_LG_)
print("stdmin_LG_:", stdmin_LG_)
print("meanmax_LG_:", meanmax_LG_)
print("meanmin_LG_:", meanmin_LG_)


# camera

mean_value = _Camera['KVI2'].mean()
std_deviation = _Camera['KVI2'].std()
minimum = _Camera['KVI2'].min()
maximum = _Camera['KVI2'].max()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)
print("Minimum:", minimum)
print("Maximum:", maximum)

# normalization
N_Camera = _Camera.copy()

scaler = MinMaxScaler()
N_Camera['prices.amountMax'] = scaler.fit_transform(N_Camera[['prices.amountMax']])
N_Camera['prices.amountMin'] = scaler.fit_transform(N_Camera[['prices.amountMin']])


# remove outlier
Q1_max = N_Camera['prices.amountMax'].quantile(0.25)
Q3_max = N_Camera['prices.amountMax'].quantile(0.75)
IQR_max = Q3_max - Q1_max
lower_bound_max = Q1_max - 1.5 * IQR_max
upper_bound_max = Q3_max + 1.5 * IQR_max
outliers_max = (N_Camera['prices.amountMax'] < lower_bound_max) | (N_Camera['prices.amountMax'] > upper_bound_max)

num_outliers_max = outliers_max.sum()
print("Number of outliers:", num_outliers_max)

total_data_points = len(N_Camera)
percentage_outliers_max = (num_outliers_max / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_max, "%")


Q1_min = N_Camera['prices.amountMin'].quantile(0.25)
Q3_min = N_Camera['prices.amountMin'].quantile(0.75)
IQR_min = Q3_min - Q1_min
lower_bound_min = Q1_min - 1.5 * IQR_min
upper_bound_min = Q3_min + 1.5 * IQR_min
outliers_min = (N_Camera['prices.amountMin'] < lower_bound_min) | (N_Camera['prices.amountMin'] > upper_bound_min)

num_outliers_min = outliers_min.sum()
print("Number of outliers:", num_outliers_min)

total_data_points = len(N_Camera)
percentage_outliers_min = (num_outliers_min / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_min, "%")


combined_outliers = outliers_max | outliers_min

N_Camera_no_outliers = N_Camera[~combined_outliers]

plt.boxplot(N_Camera_no_outliers['prices.amountMax'])
plt.show


# check and compare mean values of main competitors in a general scale

meanmax_Sony_ = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Sony') & (N_Camera_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_Sony_ = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Sony') & (N_Camera_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_Sony_ = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Sony') & (N_Camera_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_Sony_ = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Sony') & (N_Camera_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_Sony_:", stdmax_Sony_)
print("stdmin_Sony_:", stdmin_Sony_)
print("meanmax_Sony_:", meanmax_Sony_)
print("meanmin_Sony_:", meanmin_Sony_)


meanmax_Canon_ = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Canon') & (N_Camera_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_Canon_ = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Canon') & (N_Camera_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_Canon_ = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Canon') & (N_Camera_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_Canon_ = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Canon') & (N_Camera_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_Canon_:", stdmax_Canon_)
print("stdmin_Canon_:", stdmin_Canon_)
print("meanmax_Canon_:", meanmax_Canon_)
print("meanmin_Canon_:", meanmin_Canon_)


meanmax_Nikon_ = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Nikon') & (N_Camera_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_Nikon_ = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Nikon') & (N_Camera_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_Nikon_ = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Nikon') & (N_Camera_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_Nikon_ = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Nikon') & (N_Camera_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_Nikon_:", stdmax_Nikon_)
print("stdmin_Nikon_:", stdmin_Nikon_)
print("meanmax_Nikon_:", meanmax_Nikon_)
print("meanmin_Nikon_:", meanmin_Nikon_)

meanmax_Fujifilm_ = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Fujifilm') & (N_Camera_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_Fujifilm_ = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Fujifilm') & (N_Camera_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_Fujifilm_ = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Fujifilm') & (N_Camera_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_Fujifilm_ = N_Camera_no_outliers[(N_Camera_no_outliers['brand'] == 'Fujifilm') & (N_Camera_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_Fujifilm_:", stdmax_Fujifilm_)
print("stdmin_Fujifilm_:", stdmin_Fujifilm_)
print("meanmax_Fujifilm_:", meanmax_Fujifilm_)
print("meanmin_Fujifilm_:", meanmin_Fujifilm_)



# laptop

mean_value = _Laptop['KVI2'].mean()
std_deviation = _Laptop['KVI2'].std()
minimum = _Laptop['KVI2'].min()
maximum = _Laptop['KVI2'].max()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)
print("Minimum:", minimum)
print("Maximum:", maximum)

# normalization
N_Laptop = _Laptop.copy()

scaler = MinMaxScaler()
N_Laptop['prices.amountMax'] = scaler.fit_transform(N_Laptop[['prices.amountMax']])
N_Laptop['prices.amountMin'] = scaler.fit_transform(N_Laptop[['prices.amountMin']])


# remove outlier
Q1_max = N_Laptop['prices.amountMax'].quantile(0.25)
Q3_max = N_Laptop['prices.amountMax'].quantile(0.75)
IQR_max = Q3_max - Q1_max
lower_bound_max = Q1_max - 1.5 * IQR_max
upper_bound_max = Q3_max + 1.5 * IQR_max
outliers_max = (N_Laptop['prices.amountMax'] < lower_bound_max) | (N_Laptop['prices.amountMax'] > upper_bound_max)

num_outliers_max = outliers_max.sum()
print("Number of outliers:", num_outliers_max)

total_data_points = len(N_Laptop)
percentage_outliers_max = (num_outliers_max / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_max, "%")


Q1_min = N_Laptop['prices.amountMin'].quantile(0.25)
Q3_min = N_Laptop['prices.amountMin'].quantile(0.75)
IQR_min = Q3_min - Q1_min
lower_bound_min = Q1_min - 1.5 * IQR_min
upper_bound_min = Q3_min + 1.5 * IQR_min
outliers_min = (N_Laptop['prices.amountMin'] < lower_bound_min) | (N_Laptop['prices.amountMin'] > upper_bound_min)

num_outliers_min = outliers_min.sum()
print("Number of outliers:", num_outliers_min)

total_data_points = len(N_Laptop)
percentage_outliers_min = (num_outliers_min / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_min, "%")


combined_outliers = outliers_max | outliers_min

N_Laptop_no_outliers = N_Laptop[~combined_outliers]

plt.boxplot(N_Laptop_no_outliers['prices.amountMax'])
plt.show


# check and compare mean values of main competitors in a general scale

meanmax_Apple_ = N_Laptop_no_outliers[(N_Laptop_no_outliers['brand'] == 'Apple') & (N_Laptop_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_Apple_ = N_Laptop_no_outliers[(N_Laptop_no_outliers['brand'] == 'Apple') & (N_Laptop_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_Apple_ = N_Laptop_no_outliers[(N_Laptop_no_outliers['brand'] == 'Apple') & (N_Laptop_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_Apple_ = N_Laptop_no_outliers[(N_Laptop_no_outliers['brand'] == 'Apple') & (N_Laptop_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_Apple_:", stdmax_Apple_)
print("stdmin_Apple_:", stdmin_Apple_)
print("meanmax_Apple_:", meanmax_Apple_)
print("meanmin_Apple_:", meanmin_Apple_)


meanmax_Asus_ = N_Laptop_no_outliers[(N_Laptop_no_outliers['brand'] == 'Asus') & (N_Laptop_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_Asus_ = N_Laptop_no_outliers[(N_Laptop_no_outliers['brand'] == 'Asus') & (N_Laptop_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_Asus_ = N_Laptop_no_outliers[(N_Laptop_no_outliers['brand'] == 'Asus') & (N_Laptop_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_Asus_ = N_Laptop_no_outliers[(N_Laptop_no_outliers['brand'] == 'Asus') & (N_Laptop_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_Asus_:", stdmax_Asus_)
print("stdmin_Asus_:", stdmin_Asus_)
print("meanmax_Asus_:", meanmax_Asus_)
print("meanmin_Asus_:", meanmin_Asus_)


# keyboard& mouse

mean_value = _Keyboard_Mouse['KVI2'].mean()
std_deviation = _Keyboard_Mouse['KVI2'].std()
minimum = _Keyboard_Mouse['KVI2'].min()
maximum = _Keyboard_Mouse['KVI2'].max()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)
print("Minimum:", minimum)
print("Maximum:", maximum)

# normalization
N_Keyboard_Mouse = _Keyboard_Mouse.copy()

scaler = MinMaxScaler()
N_Keyboard_Mouse['prices.amountMax'] = scaler.fit_transform(N_Keyboard_Mouse[['prices.amountMax']])
N_Keyboard_Mouse['prices.amountMin'] = scaler.fit_transform(N_Keyboard_Mouse[['prices.amountMin']])


# remove outlier
Q1_max = N_Keyboard_Mouse['prices.amountMax'].quantile(0.25)
Q3_max = N_Keyboard_Mouse['prices.amountMax'].quantile(0.75)
IQR_max = Q3_max - Q1_max
lower_bound_max = Q1_max - 1.5 * IQR_max
upper_bound_max = Q3_max + 1.5 * IQR_max
outliers_max = (N_Keyboard_Mouse['prices.amountMax'] < lower_bound_max) | (N_Keyboard_Mouse['prices.amountMax'] > upper_bound_max)

num_outliers_max = outliers_max.sum()
print("Number of outliers:", num_outliers_max)

total_data_points = len(N_Keyboard_Mouse)
percentage_outliers_max = (num_outliers_max / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_max, "%")


Q1_min = N_Keyboard_Mouse['prices.amountMin'].quantile(0.25)
Q3_min = N_Keyboard_Mouse['prices.amountMin'].quantile(0.75)
IQR_min = Q3_min - Q1_min
lower_bound_min = Q1_min - 1.5 * IQR_min
upper_bound_min = Q3_min + 1.5 * IQR_min
outliers_min = (N_Keyboard_Mouse['prices.amountMin'] < lower_bound_min) | (N_Keyboard_Mouse['prices.amountMin'] > upper_bound_min)

num_outliers_min = outliers_min.sum()
print("Number of outliers:", num_outliers_min)

total_data_points = len(N_Keyboard_Mouse)
percentage_outliers_min = (num_outliers_min / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_min, "%")


combined_outliers = outliers_max | outliers_min

N_Keyboard_Mouse_no_outliers = N_Keyboard_Mouse[~combined_outliers]

plt.boxplot(N_Keyboard_Mouse_no_outliers['prices.amountMax'])
plt.show


# check and compare mean values of main competitors in a general scale

meanmax_Logitech_ = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Logitech') & (N_Keyboard_Mouse_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_Logitech_ = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Logitech') & (N_Keyboard_Mouse_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_Logitech_ = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Logitech') & (N_Keyboard_Mouse_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_Logitech_ = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Logitech') & (N_Keyboard_Mouse_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_Logitech_:", stdmax_Logitech_)
print("stdmin_Logitech_:", stdmin_Logitech_)
print("meanmax_Logitech_:", meanmax_Logitech_)
print("meanmin_Logitech_:", meanmin_Logitech_)


meanmax_Corsair_ = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Corsair') & (N_Keyboard_Mouse_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_Corsair_ = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Corsair') & (N_Keyboard_Mouse_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_Corsair_ = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Corsair') & (N_Keyboard_Mouse_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_Corsair_ = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Corsair') & (N_Keyboard_Mouse_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_Corsair_:", stdmax_Corsair_)
print("stdmin_Corsair_:", stdmin_Corsair_)
print("meanmax_Corsair_:", meanmax_Corsair_)
print("meanmin_Corsair_:", meanmin_Corsair_)

meanmax_Razer_ = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Razer') & (N_Keyboard_Mouse_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_Razer_ = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Razer') & (N_Keyboard_Mouse_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_Razer_ = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Razer') & (N_Keyboard_Mouse_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_Razer_ = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Razer') & (N_Keyboard_Mouse_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_Razer_:", stdmax_Razer_)
print("stdmin_Razer_:", stdmin_Razer_)
print("meanmax_Razer_:", meanmax_Razer_)
print("meanmin_Razer_:", meanmin_Razer_)


meanmax_Microsoft_ = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Microsoft') & (N_Keyboard_Mouse_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].mean()
meanmin_Microsoft_ = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Microsoft') & (N_Keyboard_Mouse_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].mean()
stdmax_Microsoft_ = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Microsoft') & (N_Keyboard_Mouse_no_outliers['KVI2'] >= 4.5)]['prices.amountMax'].std()
stdmin_Microsoft_ = N_Keyboard_Mouse_no_outliers[(N_Keyboard_Mouse_no_outliers['brand'] == 'Microsoft') & (N_Keyboard_Mouse_no_outliers['KVI2'] >= 4.5)]['prices.amountMin'].std()

print("stdmax_Microsoft_:", stdmax_Microsoft_)
print("stdmin_Microsoft_:", stdmin_Microsoft_)
print("meanmax_Microsoft_:", meanmax_Microsoft_)
print("meanmin_Microsoft_:", meanmin_Microsoft_)


# drive& memory (thershold = 4)

mean_value = _Drive_Memory['KVI2'].mean()
std_deviation = _Drive_Memory['KVI2'].std()
minimum = _Drive_Memory['KVI2'].min()
maximum = _Drive_Memory['KVI2'].max()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)
print("Minimum:", minimum)
print("Maximum:", maximum)

# normalization
N_Drive_Memory = _Drive_Memory.copy()

scaler = MinMaxScaler()
N_Drive_Memory['prices.amountMax'] = scaler.fit_transform(N_Drive_Memory[['prices.amountMax']])
N_Drive_Memory['prices.amountMin'] = scaler.fit_transform(N_Drive_Memory[['prices.amountMin']])


# remove outlier
Q1_max = N_Drive_Memory['prices.amountMax'].quantile(0.25)
Q3_max = N_Drive_Memory['prices.amountMax'].quantile(0.75)
IQR_max = Q3_max - Q1_max
lower_bound_max = Q1_max - 1.5 * IQR_max
upper_bound_max = Q3_max + 1.5 * IQR_max
outliers_max = (N_Drive_Memory['prices.amountMax'] < lower_bound_max) | (N_Drive_Memory['prices.amountMax'] > upper_bound_max)

num_outliers_max = outliers_max.sum()
print("Number of outliers:", num_outliers_max)

total_data_points = len(N_Drive_Memory)
percentage_outliers_max = (num_outliers_max / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_max, "%")


Q1_min = N_Drive_Memory['prices.amountMin'].quantile(0.25)
Q3_min = N_Drive_Memory['prices.amountMin'].quantile(0.75)
IQR_min = Q3_min - Q1_min
lower_bound_min = Q1_min - 1.5 * IQR_min
upper_bound_min = Q3_min + 1.5 * IQR_min
outliers_min = (N_Drive_Memory['prices.amountMin'] < lower_bound_min) | (N_Drive_Memory['prices.amountMin'] > upper_bound_min)

num_outliers_min = outliers_min.sum()
print("Number of outliers:", num_outliers_min)

total_data_points = len(N_Drive_Memory)
percentage_outliers_min = (num_outliers_min / total_data_points) * 100
print("Percentage of outliers:", percentage_outliers_min, "%")


combined_outliers = outliers_max | outliers_min

N_Drive_Memory_no_outliers = N_Drive_Memory[~combined_outliers]

plt.boxplot(N_Drive_Memory_no_outliers['prices.amountMax'])
plt.show


# check and compare mean values of main competitors in a general scale

meanmax_SanDisk_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'SanDisk') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMax'].mean()
meanmin_SanDisk_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'SanDisk') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMin'].mean()
stdmax_SanDisk_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'SanDisk') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMax'].std()
stdmin_SanDisk_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'SanDisk') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMin'].std()

print("stdmax_SanDisk_:", stdmax_SanDisk_)
print("stdmin_SanDisk_:", stdmin_SanDisk_)
print("meanmax_SanDisk_:", meanmax_SanDisk_)
print("meanmin_SanDisk_:", meanmin_SanDisk_)


meanmax_WD_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'WD') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMax'].mean()
meanmin_WD_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'WD') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMin'].mean()
stdmax_WD_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'WD') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMax'].std()
stdmin_WD_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'WD') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMin'].std()

print("stdmax_WD_:", stdmax_WD_)
print("stdmin_WD_:", stdmin_WD_)
print("meanmax_WD_:", meanmax_WD_)
print("meanmin_WD_:", meanmin_WD_)

meanmax_Seagate_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'Seagate') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMax'].mean()
meanmin_Seagate_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'Seagate') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMin'].mean()
stdmax_Seagate_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'Seagate') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMax'].std()
stdmin_Seagate_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'Seagate') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMin'].std()

print("stdmax_Seagate_:", stdmax_Seagate_)
print("stdmin_Seagate_:", stdmin_Seagate_)
print("meanmax_Seagate_:", meanmax_Seagate_)
print("meanmin_Seagate_:", meanmin_Seagate_)

meanmax_Corsair_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'Corsair') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMax'].mean()
meanmin_Corsair_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'Corsair') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMin'].mean()
stdmax_Corsair_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'Corsair') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMax'].std()
stdmin_Corsair_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'Corsair') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMin'].std()

print("stdmax_Corsair_:", stdmax_Corsair_)
print("stdmin_Corsair_:", stdmin_Corsair_)
print("meanmax_Corsair_:", meanmax_Corsair_)
print("meanmin_Corsair_:", meanmin_Corsair_)


meanmax_Samsung_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'Samsung') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMax'].mean()
meanmin_Samsung_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'Samsung') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMin'].mean()
stdmax_Samsung_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'Samsung') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMax'].std()
stdmin_Samsung_ = N_Drive_Memory_no_outliers[(N_Drive_Memory_no_outliers['brand'] == 'Samsung') & (N_Drive_Memory_no_outliers['KVI2'] >= 4)]['prices.amountMin'].std()

print("stdmax_Samsung_:", stdmax_Samsung_)
print("stdmin_Samsung_:", stdmin_Samsung_)
print("meanmax_Samsung_:", meanmax_Samsung_)
print("meanmin_Samsung_:", meanmin_Samsung_)



### check the trend with the relation of time index


merged_df2.info()

merged_df2['duration'] = merged_df2['SdateUpdated'] - merged_df2['SdateAdded']


merged_df2['duration'] = merged_df2['duration'].dt.days

#merged_df2

correlation1 = merged_df2['KVI'].corr(merged_df2['duration'])
correlation2 = merged_df2['KVI2'].corr(merged_df2['duration'])
print(correlation1)
print(correlation2)


Clist = ['Headphone_Speaker','Drive_Memory','TV','StereoSystem','Camera','Laptop','Keyboard_Mouse']
Blist = ['Sony','Samsung','Apple','Pioneer','Yamaha','SanDisk','LG','Canon','Logitech','Corsair','WD','Nikon']



for xcd in Clist:
    tdf = merged_df2[merged_df2['CATEGORY'] == xcd]
    correlation1 = tdf['KVI'].corr(tdf['duration'])
    correlation2 = tdf['KVI2'].corr(tdf['duration'])
    print(xcd, 'KVI1 vs days:', correlation1)
    print(xcd, 'KVI2 vs days:', correlation2)
    
for xcd in Blist:
    tdf = merged_df2[merged_df2['brand'] == xcd]
    correlation1 = tdf['KVI'].corr(tdf['duration'])
    correlation2 = tdf['KVI2'].corr(tdf['duration'])
    print(xcd, 'KVI1 vs days:', correlation1)
    print(xcd, 'KVI2 vs days:', correlation2)    



































