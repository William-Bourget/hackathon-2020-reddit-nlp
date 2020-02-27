# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:28:36 2020

@author: EG66349
"""

import json
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd
import json
import matplotlib.pyplot as plt
import spacy
from spacy import displacy
import numpy as np
import warnings
from sklearn.cluster import AffinityPropagation ,KMeans

from reddit import load
from preprocessing import normalize


def create_polarity_table(list_dictionnary):
    '''
    add polarity to post 
    add embeddings

    Parameters
    ----------
    list_dictionnary : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    post_id=0
    
    for post in list_dictionnary:
        title=post["title"]
        text=post["text"]
        comments_list=[comment["body"] for comment in post["comments"]]
        
        comments=""
        for comment in comments_list:
            comments += comment + " " 
        
        full = title + " " + text  #title and text
        
        title_polarity=TextBlob(title).sentiment.polarity
        text_polarity=TextBlob(text).sentiment.polarity
        comments_polarity= TextBlob(comments).sentiment.polarity
        full_polarity=TextBlob(full).sentiment.polarity
        
        
        post["polarity"] = {"title":title_polarity,
                            "text":text_polarity,
                            "comments": comments_polarity,
                            "full":full_polarity
                            
            }
        
        
        post["vectors"]={
                            "title":nlp(title).vector,
                            "text":nlp(text).vector,
                            "comments": nlp(comments).vector,
                            "full":nlp(full).vector
            
            }
        post["id"]=post_id
        post_id+=1
        
        
        
        
    pass
    


def split_pos_neg_post(list_dictionnary,pos_theshold=0.1,neg_threhold=-0.1):
    '''
    split post based on title and text polarity

    Parameters
    ----------
    list_dictionnary : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    list_positif=[]
    list_neg=[]
    for post in list_dictionnary:
        if post["polarity"]["full"]> pos_theshold:
            list_positif.append(post)
        elif  post["polarity"]["full"]< neg_threhold:
            list_neg.append(post)
        
        
        pass
    return list_positif,list_neg
    


def create_frame_shape_data_for_cluster(list_dictionnary):
    
    vectors_feature_list=[]
    
    for post in list_dictionnary:
        vectors_feature_list.append(post["vectors"]["full"])
        
        
    return np.stack(vectors_feature_list)


def create_data_frame_for_cloud(list_dictionnary,labels):
    
    
    titles=[]
    texts=[]
    labels=list(labels)
    for post in list_dictionnary:
        titles.append(post["title"])
        texts.append(post["text"])
        
        
    
    data_set={"title":titles,
              "text":texts,
              "labels": labels}
    return pd.DataFrame(data_set)
    

def create_word_cloud(df,save_name,var_name="title"):
    # plt.clf()
    comment_words = ' '
    for val in df[var_name]: 
      
        # typecaste each val to string 
        val = str(val) 
      
        # split the value 
        tokens = val.split() 
          
        # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 
              
        for words in tokens: 
            comment_words = comment_words + words + ' '
    
    
    stopwords = set(STOPWORDS) 
 
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
   
    
    wordcloud.to_file("word_cloud/"+save_name)
  
      
   


def create_word_cloud_for_all_labels(df,labels,var_name="title"):
    for label in np.unique(labels):
        df_label=df[df["labels"]==label]
        
        create_word_cloud(df_label,"label "+str(label)+".png",var_name)
    pass



def get_closest_post_to_query(df,X,query,topn=15):
    query_vector=nlp(query).vector
    id_min=np.argsort(np.linalg.norm(query_vector-X,axis=1))
    id_min=id_min[0:topn]
    return df.iloc[id_min,:]
    
    


warnings.filterwarnings("ignore", category=DeprecationWarning) 




# add additional subreddits below or comment them out with #
subreddits = """\
#https://www.reddit.com/r/AirBnB/
#https://www.reddit.com/r/AmazonFlexDrivers/
#https://www.reddit.com/r/Etsy/
#https://www.reddit.com/r/InstacartShoppers/
#https://www.reddit.com/r/TaskRabbit
#https://www.reddit.com/r/beermoney/
#https://www.reddit.com/r/couriersofreddit
#https://www.reddit.com/r/doordash/
#https://www.reddit.com/r/freelance/
https://www.reddit.com/r/lyftdrivers/
#https://www.reddit.com/r/turo/
https://www.reddit.com/r/uberdrivers/
#https://www.reddit.com/r/ridesharedrivers/
"""

data = load(subreddits.split())

#nlp = spacy.load('en_core_web_md') # disable=['parser', 'ner'])
nlp = spacy.load('en_core_web_md-2.2.5/')







tiny_data=data

create_polarity_table(tiny_data)
pos_post,neg_post=split_pos_neg_post(tiny_data,pos_theshold=0.8,
                                     neg_threhold=0.8
                                     )


chosen_data=tiny_data


X=create_frame_shape_data_for_cluster(chosen_data)

clustering = KMeans(n_clusters=12).fit(X)
labels=clustering.labels_



df=create_data_frame_for_cloud(chosen_data,labels)

#word cloud 
#create_word_cloud_for_all_labels(df,labels,"title")


querry="financial company"
top_similar=get_closest_post_to_query(df,X,querry)
top_similar.to_excel("top_similar "+querry +".xls")
