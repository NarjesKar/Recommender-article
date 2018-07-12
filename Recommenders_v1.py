import numpy as np
import pandas as pd
import scipy
import math
import random
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import utils as u 
### Add some others

            
            
class content_relatedness_doc2vec_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.doc_tag = None
        self.cosine_similarities = None
        self.user_items_strength = None
        
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id
    def get_docs(self):
        unique_train_data = self.train_data.drop_duplicates('contentId').reset_index()
        self.doc_tag = unique_train_data['contentId'].tolist()
    def get_user_interaction(self,user_id):
        user_interactions = self.train_data[self.train_data['personId'] == user_id]
        user_interactions = ((user_interactions.groupby('contentId'))['eventStrength'].sum()).reset_index()
        user_items_tags = (user_interactions['contentId']).tolist() # Get the Ids of each item for the given user             
        user_items_strength = (user_interactions['eventStrength']).tolist() # Get the weights of each item for the given user 
        return user_items_tags , user_items_strength

    def tag_docs(self):  ## This method could be used only in retraining the model or adding new items
        unique_train_data = self.train_data.drop_duplicates('contentId')
        textes = unique_train_data['text'].tolist()
        titles = unique_train_data['title'].tolist()
        doc_tag = unique_train_data['contentId'].tolist()
        self.doc_tag = doc_tag
        #
        print("Building the tagged Documents")
        docs = u.Tag_texts(textes,titles,doc_tag)
        return docs
    def set_cosine_proximity(self,mat):
        self.cosine_similarities = mat
        
              
    def build_user_profile(self,model,topn,user_id):

        user_items_tags , user_items_strength = self.get_user_interaction(user_id)        
        #other_items_tags = [item for item in self.doc_tag if item not in user_items_tags]
        ## Weighted by user pereferances
        sims = []
        for it in user_items_tags:
            sims.extend(model.docvecs.most_similar([str(it)],topn = topn))
            
        items = []
        scores = []
        for s in sims:
            if(int(s[0]) not in user_items_tags):
                if(int(s[0]) in items):
                    ind = items.index(int(s[0]))
                    scores[ind] = scores[ind] + s[1]
                    #counts[ind] = counts[ind] + 1
                else:
                    items.append(int(s[0]))
                    scores.append(s[1])
        return items , scores
    
    def recommend(self,user_id,model,top,topn):
        print("Start building user profile")
        items , scores = self.build_user_profile(model,topn,user_id)
        print("Done!")
        df_contents = pd.DataFrame(np.asarray(items) , columns = ['contentId'])
        df_scores = pd.DataFrame(np.asarray(scores) , columns = ['score'])
        scored_itmes_df = pd.concat([df_scores,df_contents],axis=1, ignore_index=False)
        result = pd.merge(scored_itmes_df, self.train_data,how='inner', on=['contentId'])
        result = ((result[['title','contentId','score']]).drop_duplicates('contentId'))
        result = result.drop_duplicates('title')
        
        recommended_df = result.sort_values('score',ascending=False)
        
        return recommended_df.head(top)