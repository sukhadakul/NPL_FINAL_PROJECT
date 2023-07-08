#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import itertools
import seaborn as sns 


# In[2]:


import warnings
warnings.filterwarnings('ignore') 


# In[3]:


fake=pd.read_csv('Fake.csv',encoding='latin-1')
true=pd.read_csv('True1 (1).csv',encoding='latin-1',error_bad_lines=False)  


# In[26]:


# Concatenate the Unnamed 4 and 5 datas in text column
drop_cols = fake.columns[4:]
fake= fake.drop(columns=drop_cols) 
#Duplicated rows
fake[fake.duplicated(keep=False)]  
# Drop duplicated records
fake.drop_duplicates(inplace=True)  


# In[45]:


# Convert date to datetime format

fake['date'] = pd.to_datetime(fake['date'], errors = 'coerce')
true['date'] = pd.to_datetime(true['date'], errors = 'coerce')
fake = fake.dropna()  


# In[46]:


# Extract the year from the date column

fake['year'] = fake['date'].dt.year 
true['year'] = true['date'].dt.year


# In[47]:


fake['year'] = fake['date'].dt.year.fillna(0).astype(int)
true['year'] = true['date'].dt.year.fillna(0).astype(int) 


# In[49]:


fake.year.unique()  


# In[50]:


true.year.unique()  


# In[ ]:





# In[ ]:





# In[27]:


fake.info() 


# In[28]:


true.info() 


# so we can see that all the data in our dataset is catagorical
# NOTE: You can see that all of the datetime related columns are not currently in datetime format. We will need to convert these later.

# In[29]:


fake.shape 


# In[30]:


true.shape 


# In[31]:


def missing_data(df):
    """
    Objective
    ----------
    it shows the missing data in each column with 
    total missing values, percentage of missing value and
    its data type in descending order.
    
    parameters
    ----------
    df: pandas dataframe
        input data frame 
    
    returns
    ----------
    missing_data: output data frame(pandas dataframe)
    
    """
    
    total = df.isna().sum().sort_values(ascending=False)
    percent = round((df.isnull().sum()/df.isna().count()  * 100).sort_values(ascending=False))
    data_type = df.dtypes
    missing_data = pd.concat([total,percent,data_type],axis=1,keys=['Total','Percent','Data_Type']).sort_values("Total", axis = 0, ascending = False)
    
    return missing_data 


# In[32]:


missing_data(true) 


# In[33]:


missing_data(fake) 


# we can clearly conclude that we don't have any missing data in our dataset .... GOOD to GO!!
# 
# Lets Check For Duplicates in our Dataset

# In[34]:


def drop_duplicates(df):
    """
    Objective
    ----------
    Drop duplicates rows in data frame except for the first occurrence.
    
    parameters
    ----------
    df: pandas dataframe
        input data frame 
        
    returns
    ----------
    dataframe with all unique rows
    """
        
    try:
        dr = df.duplicated().value_counts()
        print("[INFO] Dropping {} duplicates records...".format(dr))
        f_df = df.drop_duplicates(keep="first")
        
        return f_df
    except KeyError:
        print("[INFO] No duplicates records found")
        return df 


# In[35]:


true=drop_duplicates(true) 


# In[36]:


fake=drop_duplicates(fake) 


# In[37]:


true.shape 


# In[38]:


fake.shape 


# #Previously we had :                                                   After Removing Duplicates we have :
# 
# fake dataset have 23481 ROWS and 4 COLUMNS                            fake dataset have 23478 ROWS and 4 COLUMNS
# 
# true dataset have 21416 ROWS and 4 COLUMNS                            true dataset have 21210 ROWS and 4 COLUMNS  

# # Statistics View

# Now let's look at some statistics about the datasets 

# In[39]:


fake.describe() 


# In[40]:


true.describe()


# # Data visualization

# In[41]:


fig = plt.figure()
sns.set(rc={'figure.figsize':(3.7,8.27)})
ax1 = fig.add_subplot(2,1,1) 
sns.countplot(data = true, x = 'subject', ax = ax1) 


# After looking into the visualtion of different subject of fake news and comparing it with subjects of true news we can figure out the following things-
# 
# 1)we have only 2 subjects in true ('politicsNews', 'worldnews') while fake have 6 subjects ('News', 'politics', 'Government News', 'left-news', 'US_News', 'Middle-east) .
# 
# since we are dealing with supervised machine learing model our ulimate goal should always be to make our model as simple as possible
# with this goal we can apply some feature engineering into our fake dataset where we can merge 'News', 'left-news', 'US_News', 'Middle-east as 'worldnews' and merge 'politics', 'Government News'as 'politicsNews'

# In[43]:


true.subject.unique() 


# In[51]:


# Count of fake_news by subject
plt.figure(figsize=(15, 5))
ax1 =sns.countplot(x='subject', data=fake, saturation = 1.5)
plt.title('Count of Fake News by Subject')

for p in ax1.patches:
    ax1.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='center')
plt.show()

# Count of true_news by subject
plt.figure(figsize=(5, 3))
plt.rcParams['font.size'] = 7
ax2 = sns.countplot(x='subject', data=true, saturation = 1.5)
plt.title('Count of True News by Subject')
for p in ax2.patches:
    ax2.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='center')
plt.show() 


# In[52]:


fake.subject.unique() 


# In[53]:


fake['subject'] = fake['subject'].replace(['News'], 'worldnews')
fake['subject'] = fake['subject'].replace(['left-news'], 'worldnews')
fake['subject'] = fake['subject'].replace(['US_News'], 'worldnews')
fake['subject'] = fake['subject'].replace(['Middle-east'], 'worldnews')
fake['subject'] = fake['subject'].replace(['politics'], 'politicsNews')
fake['subject'] = fake['subject'].replace(['Government News'], 'politicsNews') 


# In[54]:


fake.subject.unique() 


# In[55]:


fig = plt.figure()
sns.set(rc={'figure.figsize':(11,8)})
ax1 = fig.add_subplot(1,2,1) 
ax2 = fig.add_subplot(1,2,2)
sns.countplot(data = fake, x = 'subject', ax = ax1,).set(xlabel='fake_Subject', ylabel='count')
sns.countplot(data = true, x = 'subject', ax = ax2).set(xlabel='True_Subject', ylabel='count') 


# Converting the date feature into months and years 

# In[58]:


from datetime import datetime
import calendar
true['date'] = pd.to_datetime(true['date'], errors='coerce')
num = true['date'].dt.month  


# In[59]:


fake['date'] = pd.to_datetime(fake['date'], errors='coerce')
num = fake['date'].dt.month 


# In[60]:


true.info() 


# In[61]:


fake.info() 


# In[62]:


fake['Month'] = fake.date.apply(lambda x:x.month)
fake['Year'] = fake.date.apply(lambda x:x.year) 


# In[63]:


true['Month'] = true.date.apply(lambda x:x.month)
true['Year'] = true.date.apply(lambda x:x.year) 


# In[64]:


fake.head()


# In[65]:


del fake['date']
del true['date'] 


# In[66]:


true.head() 


# In[67]:


fake.head() 


# In[68]:


fig = plt.figure()
sns.set(rc={'figure.figsize':(11,8)})
ax1 = fig.add_subplot(1,2,1) 
ax2 = fig.add_subplot(1,2,2)
sns.countplot(data = fake, x = 'subject', hue='Year',ax = ax1,).set(xlabel='fake_Subject', ylabel='count')
sns.countplot(data = true, x = 'subject',hue='Year', ax = ax2).set(xlabel='True_Subject', ylabel='count') 


# We can see that most of the true news revolvs around 2017 and 2016 and more of that in 2017 we have most of the true world news

# # Data Cleaning
# Now we are moving ahead with cleaning our data
# 
# Removal of HTML Contents
# 
# Removal of Punctuation Marks and Special Characters
# 
# Removal of Stopwords
# 
# Lemmatization 

# First of all we are mearging our tile,subject with text in to a single column 'text' to continue with our data cleaning process 

# In[69]:


from bs4 import BeautifulSoup
import re
import nltk
import string
from nltk.corpus import stopwords  


# In[70]:


def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()
def remove_punctuations(text):
    return re.sub('\[[^]]*\]', '', text)
def remove_characters(text):
    return re.sub("[^a-zA-Z]"," ",text) 


# In[71]:


def cleaning(text):
    text = remove_html(text)
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("[0-9" "]+"," ",text)
    text = re.sub('[‘’“”…]', '', text)
    return text

clean = lambda x: cleaning(x) 


# In[72]:


true.head() 


# In[73]:


fake.head()


# In[74]:


fake['text']=fake['text'].apply(cleaning)


# In[75]:


fake.head() 


# In[76]:


#num of Words 
fake['num_words']=fake['text'].apply(lambda x:len(nltk.word_tokenize(x)))
fake.head()
 


# In[77]:


#num of sentence 
fake['num_sent']=fake['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
del fake['num_sent'] 


# In[78]:


#Word frequency for fake dataset
freq = pd.Series(' '.join(fake['text']).split()).value_counts()[:20] # for top 50
freq


# In[79]:


#removing stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
fake['text'] = fake['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[80]:


#word frequency after removal of stopwords in fake dataset
freq_fk = pd.Series(' '.join(fake['text']).split()).value_counts()[:20] # for top 50
freq_fk 


# In[81]:


#Lets perfom lamatization into entire dataset
def lemmatize_words(text):
        lemma = nltk.WordNetLemmatizer()
        words = text.split()
        words = [lemma.lemmatize(word,pos='v') for word in words]
        return ' '.join(words) 


# In[82]:


fake['text'] = fake['text'].apply(lemmatize_words) 


# In[83]:


#word frequency after lamatization in fake dataset
freq_fk = pd.Series(' '.join(fake['text']).split()).value_counts()[:100] # for top 100
freq_fk 


# # Now as per our analysis lets add the top 100 most frequently occuring common words into stop words

# # FAKE NEWS 

# In[86]:


new_stopwords = ['trump','say','president','people','go','make','state','would','one','us','get','obama','clinton','time'] 


# In[87]:


stop.extend(new_stopwords) 


# In[88]:


stop 


# In[89]:


fake['text'] = fake['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop)) 


# In[90]:


stop = nltk.corpus.stopwords.words('english')
stop.extend(new_stopwords) 


# In[91]:


fake['text'] 


# # TRUE DATASET
#  

# In[92]:


true['text']=true['text'].apply(cleaning) 


# In[93]:


#Word frequency for true dataset
freq = pd.Series(' '.join(true['text']).split()).value_counts()[:20] # for top 50
freq


# In[94]:


#removing stopwords
stop = stopwords.words('english')
true['text'] = true['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[95]:


#num of Words 
true['num_words']=true['text'].apply(lambda x:len(nltk.word_tokenize(x)))
true.head()
 


# In[97]:


#word frequency after removal of stopwords in fake dataset
freq_tr = pd.Series(' '.join(true['text']).split()).value_counts()[:20] # for top 20
freq_tr 


# In[98]:


#Lets Perform Lamatization
true['text'] = true['text'].apply(lemmatize_words) 


# In[99]:


#word frequency after lamatization in true dataset
freq_tr = pd.Series(' '.join(true['text']).split()).value_counts()[:100] # for top 100
freq_tr 


# In[100]:


true['text'] 


# In[101]:


plt.figure(figsize=(12,6))
sns.histplot(fake['num_words'])
sns.histplot(true['num_words'],color='red')


# # We can see that in true news we have more words in comparision to fake news

# Creating a dataframe 'data'
# Adding an Target vaeriable into our data frame assigning 1 to true and 0 to fake

# In[102]:


#add column 
true['target'] = 1
fake['target'] = 0  


# In[103]:


data=pd.concat([true,fake],ignore_index=True,sort=False) 


# In[104]:


from sklearn.utils import shuffle
data=shuffle(data)
data=data.reset_index(drop=True) 


# In[105]:


data.head() 


# # Model Building

# In[127]:


data.head() 


# In[128]:


from sklearn.feature_extraction.text import TfidfVectorizer 


# In[129]:


vectorizer = TfidfVectorizer(stop_words='english',max_features=3000, max_df =1.0, smooth_idf=True) #keep top 3000 words
#doc_vec = vectorizer.fit_transform(data["text"])
#names_features = vectorizer.get_feature_names_out()
x= vectorizer.fit_transform(data["text"]).toarray() 


# In[130]:


x 


# In[160]:


)


# In[161]:


x 


# In[134]:


y=data['target'] 


# In[135]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2) 


# In[136]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 


# In[163]:


rfc = RandomForestClassifier(n_estimators=50, random_state=2) 
rfc 


# In[138]:


rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(precision_score(y_test,y_pred)) 


# In[139]:


from sklearn.metrics import classification_report 


# In[140]:


print("Classification report:")
print(classification_report(y_test,y_pred)) 


# In[141]:


import joblib
filename = "model.joblib"
joblib.dump(rfc,filename) 


# In[142]:


tfidf_obj = vectorizer.fit(data['text'])
joblib.dump(tfidf_obj, 'tf-idf.joblib') 


# In[143]:


tf_idf_converter = joblib.load("tf-idf.joblib")
classifier= joblib.load("model.joblib") 


# In[151]:


sent = ["WEST PALM BEACH, Fla (Reuters) - President Donald Trump said on Thursday he believes he will be fairly treated in a special counsel investigation into Russian meddling in the U.S. presidential election, but said he did not know how long the probe would last. The federal investigation has hung over Trumps White House since he took office almost a year ago, and some Trump allies have in recent weeks accused the team of Justice Department Special Counsel Robert Mueller of being biased against the Republican president. But in an interview with the New York Times, Trump appeared to shrug off concerns about the investigation, which was prompted by U.S. intelligence agencies conclusion that Russia tried to help Trump defeat Democrat Hillary Clinton by hacking and releasing embarrassing emails and disseminating propaganda. Theres been no collusion. But I think hes going to be fair,Trump said in what the Times described as a 30-minute impromptu interview at his golf club in West Palm Beach, Florida. Mueller has charged four Trump associates in his investigation. Russia has denied interfering in the U.S. election. U.S. Deputy Attorney General Rod Rosenstein said this month that he was not aware of any impropriety by Muellers team. Trumps lawyers have been saying for weeks that they had expected the Mueller investigation to wrap up quickly, possibly by the end of 2017. Mueller has not commented on how long it will last. Trump told the Times that he did not know how long the investigation would take. Timing-wise, I cant tell you. I just dont know,he said. Trump said he thought a prolonged probe makes the country look badbut said it has energized his core supporters. What its done is, its really angered the base and made the base stronger. My base is strong than its ever been,he said. The interview was a rare break in Trumps Christmas vacation in Florida. He has golfed each day aside from Christmas Day, and mainly kept a low profile, apart from the occasional flurry of tweets. He spent one day golfing with Republican Senator David Perdue from Georgia, who has pushed legislation to cap immigration numbers, and had dinner on Thursday with Commerce Secretary Wilbur Ross, an international trade hawk. Trump told the Times he hoped to work with Democrats in the U.S. Congress on a spending plan to fix roads and other infrastructure, and on protections for a group of undocumented immigrants who were brought to the United States as children. Trump spoke about trade issues, saying he had backed off his hard line on Chinese trade practices in the hope that Beijing would do more to pressure North Korea to end its nuclear and missile testing program. He said he had been disappointed in the results. He also complained about the North American Free Trade Agreement (NAFTA), which his administration is attempting to renegotiate in talks with Mexico and Canada. Trump said Canadian Prime Minister Justin Trudeau had played down the importance of Canadian oil and lumber exports to the United States when looking at the balance of trade between the two countries. If I dont make the right deal, Ill terminate NAFTA in two seconds. But were doing pretty good,Trump said."]


# In[152]:


prediction = classifier.predict(tf_idf_converter.transform(sent)) 


# In[153]:


print(prediction)
if (prediction[0]==0):
    print('The news is FAKE')
else:
    print('The news is True') 


# In[165]:


#prediction=rfc.predict()
    
#print(prediction)
#if (prediction[0]==0):
    #print('The news is FAKE')
#else:
    #print('The news is True') 


# In[166]:


import pickle
pickle_out=open("rfc.pkl","wb")
pickle.dump(rfc,pickle_out)
pickle_out.close()


# In[167]:


pickle_out=open("classifier.pkl","wb")
pickle.dump(classifier,pickle_out)
pickle_out.close() 


# In[168]:


pickle_out=open("tf_idf_converter.pkl","wb")
pickle.dump(tf_idf_converter,pickle_out)
pickle_out.close()


# In[ ]:




