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

#Class for Popularity based Recommender System model
class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None
        
    #Create the popularity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        #Get a count of user_ids for each unique song as recommendation score
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns = {'personId': 'score'},inplace=True)
    
        #Sort the songs based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])
    
        #Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        
        #Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)

    #Use the popularity based recommender system model to
    #make recommendations
    def recommend(self, user_id):    
        user_recommendations = self.popularity_recommendations
        
        #Add user_id column for which the recommendations are being generated
        user_recommendations['personId'] = user_id
    
        #Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        
        return user_recommendations
    

#Class for Item similarity based Recommender System model
class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
        
    #Get unique items (songs) corresponding to a given user
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        
        return user_items
        
    #Get unique users for a given item (song)
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
            
        return item_users
        
    #Get unique items (songs) in the training data
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())
            
        return all_items
        
    #Construct cooccurence matrix
    def construct_cooccurence_matrix(self, user_articles, all_articles):
            
        ####################################
        #Get users for all songs in user_articles.
        ####################################
        user_articles_users = []        
        for i in range(0, len(user_articles)):
            user_articles_users.append(self.get_item_users(user_articles[i]))
            
        ###############################################
        #Initialize the item cooccurence matrix of size 
        #len(user_articles) X len(songs)
        ###############################################
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_articles), len(all_articles))), float)
           
        #############################################################
        #Calculate similarity between user songs and all unique songs
        #in the training data
        #############################################################
        for i in range(0,len(all_articles)):
            #Calculate unique listeners (users) of song (item) i
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_articles[i]]
            users_i = set(songs_i_data[self.user_id].unique())
            
            for j in range(0,len(user_articles)):       
                    
                #Get unique listeners (users) of song (item) j
                users_j = user_articles_users[j]
                    
                #Calculate intersection of listeners of songs i and j
                users_intersection = users_i.intersection(users_j)
                
                #Calculate cooccurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    #Calculate union of listeners of songs i and j
                    users_union = users_i.union(users_j)
                    
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
                    
        
        return cooccurence_matrix

    
    #Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, user, cooccurence_matrix, all_articles, user_articles):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        
        #Calculate a weighted average of the scores in cooccurence matrix for all user songs.
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        #Sort the indices of user_sim_scores based upon their value
        #Also maintain the corresponding score
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
    
        #Create a dataframe from the following
        columns = ['personId', 'contentId', 'score', 'rank']
        #index = np.arange(1) # array of numbers for the number of samples
        df = pd.DataFrame(columns=columns)
        df['contentId'] = df['contentId'].apply(int)
        #Fill the dataframe with top 10 item based recommendations
        rank = 1 
        
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_articles[sort_index[i][1]] not in user_articles and rank <= 10:
                df.loc[len(df)]=[user,int(all_articles[sort_index[i][1]]),sort_index[i][0],rank]
                rank = rank+1
        
        #Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
 
    #Create the item similarity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    #Use the item similarity based recommender system model to
    #make recommendations
    def recommend(self, user):
        
        ########################################
        #A. Get all unique songs for this user
        ########################################
        user_articles = self.get_user_items(user)    
            
        print("No. of unique songs for the user: %d" % len(user_articles))
        
        ######################################################
        #B. Get all unique items (songs) in the training data
        ######################################################
        all_articles = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_articles))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(user_articles) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_articles, all_articles)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_articles, user_articles)
                
        return df_recommendations
    
    #Get similar items to given items
    def get_similar_items(self, item_list):
        
        user_articles = item_list
        
        ######################################################
        #B. Get all unique items (songs) in the training data
        ######################################################
        all_articles = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_articles))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(user_articles) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_articles, all_articles)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_articles, user_articles)
         
        return df_recommendations
    
class content_relatedness_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.articles_dict = None
        self.tfidf_matrix = None
        self.doc2vec_matrix = None
        self.topn = 50
        self.rev_articles_dict = None
        self.content_relatedned_recommendations = None
    def set_matrix(self,tfidf_matrix):
        self.tfidf_matrix = tfidf_matrix
        
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())           
        return all_items
    
    def get_item_profile(self,item_ids,item_id):            
        idx = item_ids.index(item_id)
        item_profile = self.tfidf_matrix[idx:idx+1]
        return item_profile 
    def get_item_profiles(self,ids,all_items):
        # all_items stands for all items in the dataset
        # ids stands only for a fraction 
        item_profiles_list = [self.get_item_profile(all_items,x) for x in ids]
        
        item_profiles = scipy.sparse.vstack(item_profiles_list)
        return item_profiles
        ## Get the Projected user Preferance
    def build_users_profile(self,user_id):
    #interactions_person_df = interactions_indexed_df.loc[person_id]
            #if(type(interactions_person_df['contentId']) == pd.core.series.Series):
        all_articles = self.get_all_items_train_data()
        user_interactions = (self.train_data[self.train_data['personId'] == user_id])
        user_item_profiles = self.get_item_profiles(list(user_interactions['contentId']),all_articles)
        user_item_strengths = (np.array(user_interactions['eventStrength'])).reshape((-1,1))
            #Weighted average of item profiles by the interactions strength
        user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
        user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
        return user_profile_norm
    def get_similar_content_items(self,user_id):
        all_articles = list(self.train_data['contentId'].unique())
        all_features = self.get_item_profiles(all_articles,all_articles)
        f_user = self.build_users_profile(user_id)
            #features_all = self.get_item_profiles()
        cosine_similarities = cosine_similarity(f_user, all_features)
        similar_indices = cosine_similarities.argsort().flatten()[-self.topn:]
        similar_items = sorted([(all_articles[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
    def recommend(self,user_id,top):
        user_interactions = self.train_data[self.train_data['personId'] == user_id].drop_duplicates('contentId') 
        print("Getting Similar items ...")
        similar_items = self.get_similar_content_items(user_id)
        print("Done !")
        similar_items_ids = [sim[0] for sim in similar_items if sim[0] not in user_interactions['contentId'].tolist()]  
        similar_items_scores = [sim[1] for sim in similar_items if sim[0] not in user_interactions['contentId'].tolist()]  
        #print(len(similar_items_ids))
        
        #print(scored_itmes_df)
        items_to_recommend = self.train_data[self.train_data['contentId'].isin(similar_items_ids)]
        items_to_recommend = (items_to_recommend[['title','contentId']]).drop_duplicates('contentId')
        
        scored_itmes_df = pd.DataFrame(np.asarray(similar_items_scores), columns = ['score'] ,index = items_to_recommend.index)
        #print(items_to_recommend)
        #print('--------------------' , len(items_to_recommend))
        #print(scored_itmes_df)
        recommended_df = pd.concat([items_to_recommend, scored_itmes_df],axis=1, ignore_index=False)

        return recommended_df.head(top)
            
            
            
class content_relatedness_doc2vec_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.doc_tag = None
        self.cosine_similarities = None
        self.user_items_strength = None
        
    #Create the popularity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id
        
    def add_tag_column(self):
        items_in_data = self.train_data['contentId']
        train_items = (items_in_data.unique()).tolist()
        doc_tag = ["DOC_" + str(i) for i in range(len(train_items))]
        list_items_in_data = items_in_data.tolist()
        tags_in_data = []
        for i in range(len(list_items_in_data)):
            ind = train_items.index(list_items_in_data[i])
            tags_in_data.append(doc_tag[ind])
        df_tags = pd.DataFrame(np.asarray(tags_in_data),columns = ['item_tag'], index=self.train_data.index)
        
        self.train_data = pd.concat([self.train_data, df_tags],axis=1, ignore_index=False)
    def tag_docs(self):
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
        
    def construct_matrix(self,model,user_id): # user_id
        unique_train_data = self.train_data.drop_duplicates('contentId')
        self.doc_tag = unique_train_data['contentId'].tolist()
        
        
        user_interactions = self.train_data[self.train_data['personId'] == user_id]
        user_interactions = ((user_interactions.groupby('contentId'))['eventStrength'].sum()).reset_index()
        user_items_tags = (user_interactions['contentId']).tolist() # Get the weights of each item for the given user
              
        user_items_strength = (user_interactions['eventStrength']).tolist()
        other_items_tags = [item for item in self.doc_tag if item not in user_items_tags]
        ## 
        print("Start Building the similarity matrix")
        cosine_similarities = []
        for it_i in user_items_tags: # user_items_tags
            cos = []
            for it_j in self.doc_tag:
                cos.append(cosine_similarity(model.docvecs[str(it_i)].reshape(1, -1), model.docvecs[str(it_j)].reshape(1, -1)))
            cosine_similarities.append(cos)
        cosine_similarities = np.asarray(cosine_similarities)
        sh = (cosine_similarities.shape[0],cosine_similarities.shape[1])
        print("Done !")
        self.cosine_similarities = cosine_similarities.reshape(sh)
        self.user_items_strength = user_items_strength
        return self.cosine_similarities
        
              
    def build_user_profile(self,topn):
        #user_interactions = self.train_data[self.train_data['personId'] == user_id]
        #user_interactions = ((user_interactions.groupby('item_tag'))['eventStrength'].sum()).reset_index()
        #user_items_tags = (user_interactions['item_tag']).tolist() # Get the weights of each item for the given user
              
        #user_items_strength = (user_interactions['eventStrength']).tolist()
        #other_items_tags = [item for item in self.doc_tag if item not in user_items_tags]
        ## Weighted by user pereferances
        #user_sim_scores = np.multiply(self.cosine_similarities,np.asarray(self.user_items_strength).reshape((-1,1))).sum(axis=0)/float(self.cosine_similarities.shape[0])
        user_sim_scores =self.cosine_similarities.sum(axis=0)/float(self.cosine_similarities.shape[0])
        similar_indices = user_sim_scores.argsort().flatten()[-topn:]
        similar_items = sorted([(self.doc_tag[i], user_sim_scores[i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
    
    def recommend(self,user_id,top):
        user_interactions = self.train_data[self.train_data['personId'] == user_id].drop_duplicates('contentId') 
        print("Getting Similar items ...")
        similar_items = self.build_user_profile(200)
        print("Done !")
        similar_items_ids = [sim[0] for sim in similar_items if sim[0] not in user_interactions['contentId'].tolist()]  
        similar_items_scores = [sim[1] for sim in similar_items if sim[0] not in user_interactions['contentId'].tolist()]  
        #print(len(similar_items_ids))
        
        #print(scored_itmes_df)
        items_to_recommend = self.train_data[self.train_data['contentId'].isin(similar_items_ids)]
        items_to_recommend = (items_to_recommend[['title','contentId']]).drop_duplicates('contentId')
        
        scored_itmes_df = pd.DataFrame(np.asarray(similar_items_scores), columns = ['score'] ,index = items_to_recommend.index)
        #print(items_to_recommend)
        #print('--------------------' , len(items_to_recommend))
        #print(scored_itmes_df)
        recommended_df = pd.concat([items_to_recommend, scored_itmes_df],axis=1, ignore_index=False)

        return recommended_df.head(top)