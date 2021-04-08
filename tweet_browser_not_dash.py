
#### Tweet Browser but not in Dash

import os

import plotly.express as px
import pandas as pd
import numpy as np
import base64
import io
import json
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse.csc import csc_matrix
from sklearn.decomposition import TruncatedSVD

# for pre-processing
import string
import nltk
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords

import sklearn.feature_extraction.text # tfidf
import umap.umap_ as umap
import textwrap # hover text on dimension reduction/clustering plot

# clustering options
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import hdbscan


# only for spyder
import plotly.io as pio
pio.renderers.default = 'browser'


#############################################################
##########   Analysis Functions   ###########################
#############################################################

# make document-word matrix from allMessages
def make_full_docWordMatrix(allMessages_json):
    if allMessages_json is not None:
        # de-json-ify cleaned tweets
        allMessages = pd.read_json(allMessages_json)
        cleanedTweets = allMessages['cleaned']
        # create document-word matrix
        vectorizer = CountVectorizer(strip_accents='unicode', min_df=2, binary=False)
        docWordMatrix_orig = vectorizer.fit_transform(cleanedTweets)
        docWordMatrix_orig = docWordMatrix_orig.astype(dtype='float64')
        # save as json
        rows_orig, cols_orig = docWordMatrix_orig.nonzero()
        data_orig = docWordMatrix_orig.data
        docWordMatrix_orig_json = json.dumps({'rows_orig':rows_orig.tolist(), 'cols_orig':cols_orig.tolist(),
            'data_orig':data_orig.tolist(), 'dims_orig':[docWordMatrix_orig.shape[0], docWordMatrix_orig.shape[1]],
            'feature_names':vectorizer.get_feature_names(), 'indices':allMessages.index.tolist()})
        return docWordMatrix_orig_json

# subset word-document matrix by words
def subset_docWordMatrix(button, docWordMatrix_orig_json, removeWords, keepWords):
    if docWordMatrix_orig_json:
        json_data = json.loads(docWordMatrix_orig_json)
        data = json_data['data_orig']
        rows = json_data['rows_orig']
        cols = json_data['cols_orig']
        dims = json_data['dims_orig']
        feature_names = json_data['feature_names']
        indices = json_data['indices']
        docWordMatrix_update = csc_matrix((data, (rows, cols)), shape=(dims[0], dims[1]))

        # subset info to output
        subset_info_remove = ''; subset_info_keep = ''
        remove_rowSums = [0] * docWordMatrix_update.shape[0]
        keep_rowSums = [1] * docWordMatrix_update.shape[0]
        if removeWords is not None: # if any remove words entered
            if len(removeWords)>0:
                remove_words = removeWords.split(',')
                remove_words_cleaned = [preProcessingFcn(word) for word in remove_words]
                remove_cols = [feature_names.index(word) for word in remove_words_cleaned if word in feature_names]
                # restrict document-word matrix to columns of remove words and take the sum of the rows
                remove_rowSums = docWordMatrix_update.tocsr()[:,remove_cols].sum(axis=1)
                # add remove words to print statement
                subset_info_remove = 'Removing tweets that contain: '
                for i in range(len(remove_words)):
                    subset_info_remove += remove_words[i] + '(' + remove_words_cleaned[i] + '),'
                subset_info_remove = subset_info_remove[:-1] + '. '
        if keepWords is not None: # if any keeps words entered
            if len(keepWords)>0:
                keep_words = keepWords.split(',')
                keep_words_cleaned = [preProcessingFcn(word) for word in keep_words]
                keep_cols = [feature_names.index(word) for word in keep_words_cleaned if word in feature_names]
                # restrict document-word matrix to columns of keep words and take sum of the rows
                keep_rowSums = docWordMatrix_update.tocsr()[:,keep_cols].sum(axis=1)
                # add keep words to print statement
                subset_info_keep = 'Keeping tweets that contain: '
                for i in range(len(keep_words)):
                    subset_info_keep += keep_words[i] + '(' + keep_words_cleaned[i] + '),'
                subset_info_keep = subset_info_keep[:-1] + '.'
        # restrict to remove/keep words
        # new doc-word matrix (and indices) as rows where total keep words >0 and total remove words =0
        new_rows = [i for i in range(len(remove_rowSums)) if remove_rowSums[i]==0 and keep_rowSums[i]>=1]
        docWordMatrix_update = docWordMatrix_update.tocsr()[new_rows]
        tfidf = sklearn.feature_extraction.text.TfidfTransformer(norm='l1').fit(docWordMatrix_update)
        docWordMatrix_tfidf = tfidf.transform(docWordMatrix_update)

        # convert to json
        rows_new, cols_new = docWordMatrix_tfidf.nonzero()
        data_new = docWordMatrix_tfidf.data
        docWordMatrix_json = json.dumps({'rows_new':rows_new.tolist(), 'cols_new':cols_new.tolist(), 'data_new':data_new.tolist(),
            'dims_new':[docWordMatrix_tfidf.shape[0], docWordMatrix_tfidf.shape[1]], 'feature_names':feature_names})

        new_indices_json = json.dumps({'indices':[indices[i] for i in new_rows]})

    else:
        docWordMatrix_json = None; new_indices_json = None
        subset_info_remove = ''; subset_info_keep = ''
    
    return docWordMatrix_json, subset_info_remove, subset_info_keep, new_indices_json

# dimension reduction part 1: PCA
def PCA_docWordMatrix(docWordMatrix_json):
    if docWordMatrix_json is not None:
        json_data = json.loads(docWordMatrix_json)
        data = json_data['data_new']
        rows = json_data['rows_new']
        cols = json_data['cols_new']
        dims = json_data['dims_new']
        docWordMatrix = csc_matrix((data, (rows, cols)), shape=(dims[0], dims[1]))
        tsvd = TruncatedSVD(n_components=25)
        tsvd.fit(docWordMatrix)
        docWordMatrix_pca = tsvd.transform(docWordMatrix)
        docWordMatrix_pca_json = json.dumps(docWordMatrix_pca.tolist())
        return docWordMatrix_pca_json

# reduce to 2D using UMAP
def get_dimRed_points(docWordMatrix_pca_json, new_indices_json, dimRed_method):
    if docWordMatrix_pca_json is not None:
        docWordMatrix_pca = pd.read_json(docWordMatrix_pca_json)
        indices = json.loads(new_indices_json)['indices']
        # do dimension reduction
        if dimRed_method == 'umap':
            umap_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.0)
            proj_2d = umap_2d.fit_transform(docWordMatrix_pca)  
        # convert 2d points to json
        dimRed_df = pd.DataFrame({'coord1':proj_2d[:,0], 'coord2':proj_2d[:,1]}, index=indices)
        dimRed_points_json = dimRed_df.to_json()

        return dimRed_points_json
    
# clustering on 2D points
def make_cluster_plot(dimRed_points_json, allMessages_json, clustering_method, num_clusters, min_obs):
    dimRed_cluster_plot = {}
    if dimRed_points_json is not None:
        # get dimension reduced points
        dimRed_points = pd.read_json(dimRed_points_json)
        # get allMessages
        allMessages = pd.read_json(allMessages_json)
        # merge
        allMessages_plot = allMessages.merge(dimRed_points, how='right', left_index=True, right_index=True)
        allMessages_plot['Text'] = allMessages_plot['Message'].apply(lambda t: "<br>".join(textwrap.wrap(t)))
        # do clustering
        if clustering_method=='gmm':
            gmm = GaussianMixture(n_components=num_clusters, random_state=42).fit(allMessages_plot[['coord1', 'coord2']])
            allMessages_plot['Cluster'] = gmm.predict(allMessages_plot[['coord1', 'coord2']]).astype(str)
        if clustering_method=='k-means':
            kmeans = KMeans(init='random', n_clusters=num_clusters, random_state=42)
            allMessages_plot['Cluster'] = kmeans.fit(allMessages_plot[['coord1', 'coord2']]).labels_.astype(str)
        if clustering_method=='hdbscan':
            hdbscan_fcn = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=min_obs)
            allMessages_plot['Cluster'] = hdbscan_fcn.fit_predict(allMessages_plot[['coord1', 'coord2']]).astype(str)
        dimRed_cluster_plot = px.scatter(allMessages_plot, x='coord1', y='coord2', color='Cluster',
            hover_data=['Text'])
        dimRed_cluster_plot.update_layout(clickmode='event+select')
        # cluster info to output
        cluster_data = allMessages_plot[['Cluster']]
        cluster_data_json = cluster_data.to_json()
    return dimRed_cluster_plot, cluster_data_json

# make table of cluster info
def make_cluster_table(docWordMatrix_orig_json, cluster_data_json):
    if cluster_data_json is None:
        return None, None
    else:
        # de-json-ify doc-word matrix
        json_data = json.loads(docWordMatrix_orig_json)
        data = json_data['data_orig']
        rows = json_data['rows_orig']
        cols = json_data['cols_orig']
        dims = json_data['dims_orig']
        feature_names = json_data['feature_names']
        indices = json_data['indices']
        docWordMatrix = csc_matrix((data, (rows, cols)), shape=(dims[0], dims[1]))
        # de json-ify cluster data
        cluster_data = pd.read_json(cluster_data_json)
        # representative words and tweet IDs for each cluster
        clusters_for_table = []; topWords_data = []; prop_data = []; num_data = []
        for i in cluster_data.Cluster.unique():
            # restrict docWordMatrix to rows in cluster i
            i_indices = cluster_data.index[cluster_data['Cluster']==i].tolist()
            docWordMat_rows = [j for j in range(len(indices)) if indices[j] in i_indices]
            clusteri = docWordMatrix[docWordMat_rows, :]
            # get most common words
            colSumsi = np.squeeze(np.asarray(clusteri.sum(axis=0)))
            top5 = np.sort(colSumsi)[-5]
            topWordsi = [feature_names[j] for j in range(len(colSumsi)) if colSumsi[j]>=top5][0:5]
            # add to display data
            clusters_for_table.append(i)
            topWords_data.append(" ".join(topWordsi))
            prop_data.append(round(clusteri.shape[0]/dims[0], 3))
            num_data.append(clusteri.shape[0])
        display_df = pd.DataFrame(data={'Cluster':clusters_for_table, 'Proportion of Tweets':prop_data,
            'Number of Tweets':num_data, 'Top Stemmed Words':topWords_data})

        return display_df
    

#############################################################
###########   Inputs   ######################################
#############################################################

# directory with data
os.chdir('')

# dataset to use
dataset = ''

# subsetting
removeWords = None
keepWords = None

# clustering
clustering_method = 'gmm' # options: 'hdbscan', 'gmm', 'k-means'
num_cluster = 30 # for gmm and k-means
min_obs = 500 # for hdbscan

#############################################################
##########   Analysis   #####################################
#############################################################

# Get data
allMessages = pd.read_csv(dataset)
if type(allMessages['UniversalMessageId'][0]) is not str:
    allMessages['UniversalMessageId'] = ['twitter'+allMessages['UniversalMessageId'][i].astype(str) for i in range(allMessages.shape[0])]
allMessages = allMessages.set_index('UniversalMessageId')
allMessages = allMessages.dropna(subset=['cleaned'])
allMessages_json = allMessages.to_json()

# document-word matrix
docWordMatrix_orig_json = make_full_docWordMatrix(allMessages_json)

# subset by words
docWordMatrix_json, subset_info_remove, subset_info_keep, new_indices_json = subset_docWordMatrix('button', docWordMatrix_orig_json, removeWords, keepWords)

# reduce dimension
docWordMatrix_pca_json = PCA_docWordMatrix(docWordMatrix_json)

# reduce to 2D using UMAP
dimRed_points_json = get_dimRed_points(docWordMatrix_pca_json, new_indices_json, dimRed_method='umap')

# clustering and plot
dimRed_cluster_plot, cluster_data_json = make_cluster_plot(dimRed_points_json, allMessages_json, clustering_method, num_cluster, min_obs)
dimRed_cluster_plot.show()

# table of cluster info
cluster_table = make_cluster_table(docWordMatrix_orig_json, cluster_data_json)
print(cluster_table.to_string())

text_file = open('table.html','w')
text_file.write(cluster_table.to_html())
text_file.close()
