  
### Dash Tweet Browser

import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
# import dash_core_components as dcc
# import dash_html_components as html
from dash.dependencies import Input, Output, State
# import dash_table
from dash import dash_table
from dash_extensions import Download # download csv
from dash_extensions.snippets import send_data_frame # download csv
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import math
import base64
import io
import json
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse.csc import csc_matrix

# for uploading
import dash_uploader as du

# for pre-processing
import string
import nltk
from nltk.stem import PorterStemmer
import re
nltk.download('stopwords')
from nltk.corpus import stopwords

import sklearn.feature_extraction.text # tfidf
import umap.umap_ as umap
import textwrap # hover text on dimension reduction/clustering plot
from sklearn.decomposition import TruncatedSVD

# clustering options
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import hdbscan

# R related
import rpy2
from rpy2.robjects.packages import importr
from rpy2 import robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()

# import R's packages
dplyr = importr('dplyr', lib_loc = "/usr/local/lib/R/site-library")
reshape2 = importr('reshape2', lib_loc = "/usr/local/lib/R/site-library")

# clear 'temp' folder every week
import os, sys, time
from subprocess import call

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

app.config.suppress_callback_exceptions = True

du.configure_upload(app, folder='temp', use_upload_id = False)

## Layout

def Navbar():
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href='/index')),
            dbc.DropdownMenu(
                children=[dbc.DropdownMenuItem('Info', href='#')],
                nav=True,
                in_navbar=True,
                label="More"
            ),
        ],
        brand='Tweet Browser',
        color='primary',
        dark=True
    )
    return navbar

marginBottom = 50
marginTop = 25
marginLeft = 15
marginRight = 15

style = {'marginBottom':marginBottom, 'marginTop':marginTop, 'marginRight':marginRight, 'marginLeft':marginLeft}
style_upload = {
                "width": "75%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            }

main_layout = html.Div([

    ## Upload Data
    # html.Div([
    #     html.H2('Upload Data'),
    #     dcc.Upload(id='upload_all', children=html.Div(['Drag and Drop or ', html.A('Select File')]),
    #         style=style_upload)
    # ], style=style),
    html.Div(
        dbc.Container(
            [
                html.H2('Upload Data'),
                dcc.Markdown('''
                [Input file example (csv)](https://github.com/juejuew/Census_tweet_browser/blob/master/allCensus_sample.csv)

                Supported input file formats: CSV, TXT, TSV, XLS

                The input dataset needs to contain at least these columns:
                
                * "UniversalMessageId" : Unique ID that identifies each tweet
                * "Message" (string) : Tweet messages
                * “Day" (yyyy-mm-dd): The day each tweet was sent out
                * "State" (string) : State from where each tweet was sent out. Associated dictionary can be found on [Github](https://github.com/juejuew/Census_tweet_browser/blob/master/app_with_advanced_options.py)
                * "Sender.Followers.Count" (int) : The number of followers the sender of each tweet has

                **Note**: The tweet browser pre-processes the original messages automatically. If you have pre-processed the tweets and do not want this app to pre-process again, please create a new column that contains your pre-processed tweets and name this new column with the name "cleaned" 
            '''),
                du.Upload(id='upload_all'),
                html.H5(' Uploading or not yet ！', id='upload_status')
                ]
            )
        ),
    
    # html.Div(id='filename_info'),

    ## Subsetting
    # remove/keep words input
    html.Div([
        html.H2('Subset Data (Optional)'),
        html.I('Remove tweets containing at least one of the following words (separated by commas):'),
        dcc.Input(id='removeWords',type='text', placeholder='(Optional)'),
        html.P(),
        html.I('Keep tweets containing at least one of the following words (separated by commas):'),
        dcc.Input(id='keepWords', type='text', placeholder='(Optional)'),
        html.P(),
        html.Button(id='button', children='Submit'),
        html.P()
    ], style=style),
    
    html.Div(id='subset_info_remove'),
    html.Div(id='subset_info_keep'),

    html.Hr(), html.Hr(),

    ### Section 1: entire forest
    html.Div([html.H1('Explore Tweet Clusters')],
        style={'marginBottom':marginBottom, 'marginTop':marginTop, 'marginRight':marginRight, 'marginLeft':marginLeft,
            'text-align': 'center'}),
    ## Inputs
    html.Div([
        # default or advanced option for dimension reduction/clustering
        html.H3('Method for Finding Clusters:'), 
        dcc.RadioItems(id='default_or_advanced', options=[
                {'label': 'Default', 'value': 'default'},
                {'label': 'Advanced', 'value': 'advanced'}
            ],
            value='default',
            # labelStyle={'display': 'inline-block'}
            labelStyle={
                'display': 'inline-block',
                'margin-right': '7px',
                'font-weight': 300
                },
            style={
                'display': 'inline-block',
                'margin-left': '7px'
                }
        ),
        html.P(),
        # Advanced options
        # text explaining advanced options
        html.Div(id='advanced_options', children=[
            dcc.Markdown('''
                The advanced options allow for up to two stages of dimension reduction and choice of clustering method.

                The choices of dimension reduction methods are [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) and [UMAP](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Uniform_manifold_approximation_and_projection). 2nd Stage Dimension Reduction always reduces to 2 dimension for plotting.

                The choices of clustering methods are [Gaussian Mixture Model](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model), [K-Means](https://en.wikipedia.org/wiki/K-means_clustering), and [HDBSCAN](https://en.wikipedia.org/wiki/DBSCAN).

                The default option is dimension reduction to 25 dimensions using PCA, dimension reduction to 2 dimensions using UMAP, and clustering using GMM.
            '''),
            # dimension reduction options
            html.H5('Dimension Reduction'),
            html.H6('1st Stage Dimension Reduction'),
            # 1st stage: method and number of dimensions
            dcc.Dropdown(id='dimRed1_method', multi=False,
                options=[{'label': 'UMAP', 'value': 'umap'}, {'label': 'PCA', 'value': 'pca'}], value='pca'),
            html.Div(children='Number of dimensions:'),
            # dcc.Input(id='dimRed1_dims', type='number', debounce=True, 
            #     min=2, max=1000, step=1, value=25),
            dcc.Slider(
                id='dimRed1_dims',
                min=5,
                max=200,
                step=5,
                value=25,
                tooltip={"placement": "bottom", "always_visible": True},
                marks={
                    5: '5',
                    50: '50',
                    100: '100',
                    150: '150',
                    200: '200'
                    },),
            # html.Div(id='slider-output-container'),
            # 2nd stage: method and number of dimensions
            html.H6('2nd Stage Dimension Reduction'),
            dcc.Dropdown(id='dimRed2_method', multi=False,
                options=[{'label': 'UMAP', 'value': 'umap'}, {'label': 'PCA', 'value': 'pca'}], value='umap'), 
            html.P(),     
            # clustering
            html.H5('Clustering'),
            # methods
            dcc.Dropdown(id='clustering_method1', multi=False,
                options=[{'label': 'Gaussian Mixture Model', 'value': 'gmm'},
                    {'label': 'K-Means', 'value': 'k-means'},
                    {'label': 'HDBSCAN', 'value': 'hdbscan'}],
                value='gmm'),
            # when to perform clustering
            html.Div(children='When to perform clustering:'),
            dcc.Dropdown(id='clustering_when', multi=False,
                options=[{'label': 'Before stage 1', 'value': 'before_stage1'},
                    {'label': 'Between stages 1 and 2', 'value': 'btwn'},
                    {'label': 'After stage 2', 'value': 'after_stage2'}],
                value='after_stage2'),
            html.P()
        ]),
        # clustering options
        # for gmm and k-means: number of clusters
        html.Div(id='num_clustering_input_container', 
            children=[
                html.H6('Number of Clusters'),
                dcc.Slider(
                id='num_clusters',
                min=2,
                max=50,
                step=1,
                value=5,
                tooltip={"placement": "bottom", "always_visible": True},
                marks={
                    2: '2',
                    10: '10',
                    20: '20',
                    30: '30',
                    40: '40',
                    50: '50'
                    },)
                # dcc.Input(id='num_clusters', type='number', value=5)
            ]
        ),
        # for hdbscan: minimum number of observations in each cluster
        html.Div(id='min_size_input_container',
            children=[
                html.H6('Minimum Number of Observations in Each Cluster'),
                dcc.Input(id='min_obs', type='number', value=100)
            ]
        )       
    ], style=style),

    # 2D interactive plot of tweets
    # dcc.Graph(id='dimRed_cluster_plot'),
    dbc.Row(dbc.Col(
           dbc.Spinner(children=[dcc.Graph(id="dimRed_cluster_plot")], color="primary", type="border", fullscreen=False,),
           width={'size': 12, 'offset': 0}),
       ),
    # Table of number of tweets and top words for each cluster
    html.H4('Most common words in each cluster:'),
    html.Div(id='top_words_num_input',
            children=[
                html.H6('Number of top words'),
                dcc.Dropdown(
                    id='top_words_num',
                    clearable=False,
                    options=[{'label': i, 'value': i} for i in [5,10,15,20,25,30]],
                    value=5, 
                    placeholder='Select...',),
                ], style=dict(width='40%')),
    html.H6(''),
    html.Div([
        dash_table.DataTable(
            id='clusters_table',
            style_cell={'textAlign': 'left'},
            style_as_list_view=True,
            style_table={'minWidth':'100%', 'width': '80%'},
            sort_action='native'
        )
    ],style={'marginRight': 100, 'marginLeft': 100}),
    html.H6(''),
    html.H6(''),
    html.H6(''),
    # Dot plot of feature words for each cluster
    html.H4('Feature words in each cluster:'),
    html.Div(id='common_feature_perc_input',
            children=[
                html.H6('Remove common words'),
                dcc.Markdown('''
                choose a percentage p such that top words appears in >= p% of clusters will be removed from the feature words. '''),
                dcc.Slider(
                id='feature_perc_threshold',
                min=0,
                max=100,    
                step=1,
                value=60,
                tooltip={"placement": "bottom", "always_visible": True},
                marks={
                    0: '0',
                    10: '10',
                    20: '20',
                    30: '30',
                    40: '40',
                    50: '50',
                    60: '60',
                    70: '70',
                    80: '80',
                    90: '90',
                    100: '100',
                    },),
                ]),
    html.Div(id='words_freq_threshold', style={'display': 'none'}),
    html.Div(id='feature_words_list', style={'display': 'none'}),
    html.Div(id='feature_slider_output'),

    dcc.Markdown('''
                For the dot plot below, the dots’ color represents the average frequency for words in their corresponding clusters, and the size describes the proportion of tweets (in the corresponding cluster) that contain the words.
            '''),

    dbc.Row(dbc.Col(
           dbc.Spinner(children=[dcc.Graph(id="feature_cluster_dot_plot")], color="primary", type="border", fullscreen=False,),
           width={'size': 12, 'offset': 0}),style = style),
    # html.Div(id='words_freq_threshold'),

    html.Hr(), html.Hr(),
    
    ### Section 2: new characteristics of cluster(s) 
    html.Div([html.H1('Characteristics of Tweet Clusters')],
        style={'marginBottom':marginBottom, 'marginTop':marginTop, 'marginRight':marginRight, 'marginLeft':marginLeft,
            'text-align': 'center'}),
    
    ## inputs
    html.Div([
        # default or advanced option for dimension reduction/clustering
        html.H3('Method for Selecting Tweets:'), 
        dcc.RadioItems(id='cluster_or_manual', options=[
                {'label': 'Select by clusters', 'value': 'cluster'},
                {'label': 'Select tweets manually', 'value': 'manual'}
            ],
            value='cluster',
            # labelStyle={'display': 'inline-block'}
            labelStyle={
                'display': 'inline-block',
                'margin-right': '7px',
                'font-weight': 300
                },
            style={
                'display': 'inline-block',
                'margin-left': '7px'
                }
        ),
        html.P(),
        # Advanced options
        # text explaining advanced options
        html.Div(id='manual_select_options', children=[
            html.H2('Select tweets'),
            dcc.Markdown('''
            * To select tweets (represented by the points on the graph), click on points in the graph or use "Box Select" and "Lasso Select" tools in the graph's menu bar. 
            * To accumulate the selection data, please hold down the shift button on keyboard while selecting.
            '''),
            ## Plot to select tweets
            html.Div([
                html.Button(id='ignore',style={'display':'none'}),
                dbc.Row(dbc.Col(
                    dbc.Spinner(children=[dcc.Graph(id='dimRed_cluster_plot3')], color="primary", type="border", fullscreen=False,),
                    width={'size': 12, 'offset': 0}),
                       ),
            ]),
            
            html.Div(id='output_tweet_count'),
            
            ## which characteristic(s)
            html.H2('Tweet Set Characteristics'),
            dcc.Dropdown( 
                id = 'selected_chars',
                options=[
                    {'label': 'Frequency over Time--Raw Number', 'value': 'freq_raw'},
                    {'label': 'Frequency over Time--Normalized', 'value': 'freq_norm'}, 
                    {'label': 'Geographical Location', 'value': 'geo'},
                    {'label': 'Geographical Location--Normalized', 'value': 'geo_norm'},
                    {'label': 'Random Sample of Tweets--Simple Random Sample', 'value': 'randSamp'},
                    {'label': 'Random Sample of Tweets--Weighted by Followers', 'value': 'randSamp_follow'},
                    {'label': 'Ransom Sample of Tweets--Unique Tweets', 'value': 'randSamp_unique'}
                ], 
                multi=True
            ),
            ## output plots
            html.Div([
                # table of simple random sample of tweets
                html.Div(id='selected_randomSample_title', style={'font-weight':'bold'}),
                html.Div(id='selected_table_randomSample'),
                # table of random sample of tweets, weighted by number of followers (i.e. tweets more people are likely to see)
                html.Div(id='selected_randomSampleFollow_title', style={'font-weight':'bold'}),
                html.Div(id='selected_table_randomSampleFollow'),
                # table of random sample of unique tweets in a clusters
                html.Div(id='selected_randomSampleUnique_title', style={'font-weight':'bold'}),
                html.Div(id='selected_table_randomSampleUnique'),
                # raw number of tweets over time
                html.Div(id='selected_freq_raw_plot_container', children=[dcc.Graph(id='selected_freq_raw_plot')]), 
                # number of tweets over time, normalized (easier to compare across clusters)
                html.Div(id='selected_freq_norm_plot_container', children=[dcc.Graph(id='selected_freq_norm_plot')]),
                # maps of frequency of tweets per state
                html.Div(id='selected_geo_plot_container', children=[dcc.Graph(id='selected_plot_geo')]),
                html.Div(id='selected_geo_plot_norm_container', children=[dcc.Graph(id='selected_plot_geo_norm')])
            ], style=style),
            ## download selected tweets 
            html.Div([
                html.H4('Download csv of selected tweets'),
                html.I('Name of the csv:'),
                dcc.Input(id='selected_csv_name',type='text'),
                html.P(),
                html.Button('Download csv', id='download_button1'),
                Download(id='download1')
            ], style=style),
        ]),
        html.Div(id='cluster_select_options', children=[
           ## inputs
            html.Div([
                # which tweet cluster(s)
                html.H2('Tweet Sets'),
                dcc.Dropdown(id='unique_cluster_options', multi=True, placeholder='Select at least one cluster'),
                # which characteristic(s)
                html.H2('Tweet Set Characteristics'),
                dcc.Dropdown( 
                    id = 'cluster_chars',
                    options=[
                        {'label': 'Frequency over Time--Raw Number', 'value': 'freq_raw'},
                        {'label': 'Frequency over Time--Normalized', 'value': 'freq_norm'}, 
                        {'label': 'Geographical Location', 'value': 'geo'},
                        {'label': 'Geographical Location--Normalized', 'value': 'geo_norm'},
                        {'label': 'Random Sample of Tweets--Simple Random Sample', 'value': 'randSamp'},
                        {'label': 'Random Sample of Tweets--Weighted by Followers', 'value': 'randSamp_follow'},
                        {'label': 'Ransom Sample of Tweets--Unique Tweets', 'value': 'randSamp_unique'}
                    ], 
                    multi=True
                )
            ], style=style),
            
            ## output plots
            html.Div([
                # table of simple random sample of tweets
                html.Div(id='randomSample_title', style={'font-weight':'bold'}),
                html.Div(id='table_randomSample'),
                # table of random sample of tweets, weighted by number of followers (i.e. tweets more people are likely to see)
                html.Div(id='randomSampleFollow_title', style={'font-weight':'bold'}),
                html.Div(id='table_randomSampleFollow'),
                # table of random sample of unique tweets in a clusters
                html.Div(id='randomSampleUnique_title', style={'font-weight':'bold'}),
                html.Div(id='table_randomSampleUnique'),
                # raw number of tweets over time
                html.Div(id='freq_raw_plot_container', children=[dcc.Graph(id='freq_raw_plot')]), 
                # number of tweets over time, normalized (easier to compare across clusters)
                html.Div(id='freq_norm_plot_container', children=[dcc.Graph(id='freq_norm_plot')]),
                # maps of frequency of tweets per state
                html.Div(id='geo_plot_container', children=[dcc.Graph(id='plot_geo')]),
                html.Div(id='geo_plot_norm_container', children=[dcc.Graph(id='plot_geo_norm')])
            ], style=style),
            ## download tweet option
            html.Div([
                html.H4('Download csv of a single tweet cluster'),
                dcc.Dropdown(id='download_tweet_cluster', placeholder='Select a cluster',
                             style=dict(width='40%')),
                html.Button('Download csv', id='download_button'),
                Download(id='download')
            ], style=style),
        ]),
    ], style=style),
    
    html.Hr(), html.Hr(),

    ### Section 3: individual tweet(s) info
    html.Div([html.H1('Individual Tweet Information')],
            style={'marginBottom':marginBottom, 'marginTop':marginTop, 'marginRight':marginRight, 'marginLeft':marginLeft,
            'text-align': 'center'}),
    # html.Div([
    # 		dcc.Graph(id='dimRed_cluster_plot2'),
    # 		html.Div([html.Pre(id='click_data')])
    #  	]),

    dbc.Row(dbc.Col(
           dbc.Spinner(children=[dcc.Graph(id='dimRed_cluster_plot2'),          
               html.Div([html.Pre(id='click_data')])], color="primary", type="border", fullscreen=False,),
           width={'size': 12, 'offset': 0}),
       ),


    ### Hidden Output
    ## in Dash, all function outputs must be listed in the layout
    ## these are json files output by functions below to be used in later functions
    html.Div(id='allMessages_json', style={'display': 'none'}),
    html.Div(id='docWordMatrix_orig_json', style={'display': 'none'}),
    html.Div(id='docWordMatrix_json', style={'display':'none'}),
    html.Div(id='new_indices_json', style={'display': 'none'}),
    html.Div(id='dimRed_points_json', style={'display': 'none'}),
    html.Div(id='cluster_ids_json', style={'display': 'none'}),
    html.Div(id='selected_subset_json', style={'display': 'none'}),
])

app.layout = html.Div([Navbar(), main_layout])

# def show_upload_status(isCompleted, fileNames):
#     if isCompleted:
#        return ' Finished uploading ：'+fileNames[0]
#     return dash.no_update

# show slider value for the first dimension reduction
# @app.callback(
#     dash.dependencies.Output('slider-output-container', 'children'),
#     [dash.dependencies.Input('dimRed1_dims', 'value')])
# def update_output(value):
#     return 'You have selected "{}"'.format(value)

# this function reads in the data (copied from online)
def parse_data(filename):
    path = 'temp/' + filename
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(path, encoding = "utf-8", index_col=[0])
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(path, index_col=[0])
        elif "txt" or "tsv" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_csv(path, delimiter = "\t",encoding = "ISO-8859-1",  index_col=[0])
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])
    return df

def preProcessingFcn(tweet, removeWords=list(), stem=True, removeURL=True, removeStopwords=True, 
    removeNumbers=False, removePunctuation=True):
    """
    Cleans tweets by removing words, stemming, etc.
    """
    ps = PorterStemmer()
    tweet = tweet.lower()
    tweet = re.sub(r"\\n", " ", tweet)
    tweet = re.sub(r"&amp", " ", tweet)
    if removeURL==True:
        tweet = re.sub(r"http\S+", " ", tweet)
    if removeNumbers==True:
        tweet=  ''.join(i for i in tweet if not i.isdigit())
    if removePunctuation==True:
        for punct in string.punctuation:
            tweet = tweet.replace(punct, ' ')
    if removeStopwords==True:
        tweet = ' '.join([word for word in tweet.split() if word not in stopwords.words('english')])
    if len(removeWords)>0:
        tweet = ' '.join([word for word in tweet.split() if word not in removeWords])
    if stem==True:
        tweet = ' '.join([ps.stem(word) for word in tweet.split()])
    return tweet

# creates table--used for displaying random samples of tweets
def generate_table(dataframe, max_rows=100):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

def choose_features(cluster_table, top_words_num = 5, threshold = 60):
    perc_threshold = round(cluster_table.shape[0] * threshold / 100)
    if perc_threshold <= 0:
        perc_threshold = 2
    top_words = []
    topWords_str = 'Top ' + str(top_words_num) + ' Stemmed Words'
    for i in cluster_table[topWords_str]:
        top_words.append(i.split())
    
    flat_top_words = [item for sublist in top_words for item in sublist]
    
    from collections import Counter
    D_T_dict = dict(Counter(flat_top_words))
    
    common_words = []
    for i in range(len(D_T_dict)):
        word = list(D_T_dict.keys())[i]
        count = list(D_T_dict.values())[i]
        if count >= perc_threshold:
            common_words.append(word)
            
    for key in common_words:
        del D_T_dict[key]
        
    return perc_threshold, list(D_T_dict.keys())

def feature_table(all_words, 
                  feature_words, 
                  group_idx, 
                  DT_mtx):
    robjects.r(
        '''
        # create a function `f`
        f <- function(all_words, 
                      feature_words, 
                      group_idx, 
                      DT_mtx) {

            # Open a file to send messages to
            zz <- file("messages.Rout", open = "wt")
            # Divert messages to that file
            sink(zz, type = "message")
            message("not gonna show up in console")

            all_words = as.character(unlist(all_words))
            feature_words = as.character(unlist(feature_words))
            group_idx = as.character(unlist(group_idx))
            DT_mtx = as.data.frame(DT_mtx)
             
            colnames(DT_mtx) = all_words
            exprs_mat = reshape2::melt(DT_mtx[,feature_words])
            exprs_mat = exprs_mat %>% mutate(Tweet = rep(1:nrow(DT_mtx), length(feature_words)))
            colnames(exprs_mat) = c('Word', 'Expression', 'Tweet')
            exprs_mat$Words = as.character(exprs_mat$Word)
            
            exprs_mat$Group = group_idx
            exprs_mat = exprs_mat %>% dplyr::filter(is.na(Group) == FALSE)
            
            ExpVal = exprs_mat %>% dplyr::group_by(Group, Word) %>%
                dplyr::summarize(mean = mean(Expression),
                     percentage = sum(Expression > 0) /
                       length(Expression))
            ExpVal = ExpVal %>% group_by(Group) %>% mutate(avg_mean = mean(mean), avg_perc = mean(percentage))
            
            ExpVal$Word = factor(ExpVal$Word, levels = feature_words)
        
            return(ExpVal)
        }
        ''')
    r_f = robjects.globalenv['f']
    return(r_f(all_words, feature_words, group_idx, DT_mtx))

# state abbreviations used in map of tweet frequencies
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Washington, D.C.': 'DC',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

# state populations, from https://www.census.gov/data/tables/time-series/demo/popest/2010s-state-total.html
# need for normalizing number of tweets per person in each state
state_pops = pd.read_csv('state_populations.csv', index_col=0, header=None)
state_pops.index = [state[1:] for state in state_pops.index]
state_pops = state_pops.iloc[:,0]

# convert population to numeric 
state_pops_tmp = pd.Series([float(pop.replace(',', '')) for pop in state_pops])
state_pops_tmp.index = [state for state in state_pops.index]
state_pops = state_pops_tmp

# functions for dimension reduction: PCA and UMAP
def dimred_PCA(docWordMatrix, ndims=25):
    tsvd = TruncatedSVD(n_components=ndims)
    tsvd.fit(docWordMatrix)
    docWordMatrix_pca = tsvd.transform(docWordMatrix)
    return docWordMatrix_pca

def dimred_UMAP(matrix, ndims=2, n_neighbors=15):
    umap_2d = umap.UMAP(n_components=ndims, random_state=42, n_neighbors=n_neighbors, min_dist=0.0)
    proj_2d = umap_2d.fit_transform(matrix)
    return proj_2d

# functions for clustering
# HDBSCAN
def cluster_hdbscan(points, min_obs):
    hdbscan_fcn = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=min_obs)
    clusters = hdbscan_fcn.fit_predict(points).astype(str)
    return clusters

# Gaussian Mixure Models
def cluster_gmm(points, num_clusters):
    gmm_fcn = GaussianMixture(n_components=num_clusters, random_state=42).fit(points)
    clusters = gmm_fcn.predict(points).astype(str)
    return clusters

# K-Means
def cluster_kmeans(points, num_clusters):
    kmean_fcn = KMeans(init='random', n_clusters=num_clusters, random_state=42)
    clusters = kmean_fcn.fit(points).labels_.astype(str)
    return clusters


## Read in data
@app.callback([Output('allMessages_json', 'children'),
        Output('upload_status', 'children')],
    Input('upload_all', 'isCompleted'),
    State('upload_all', 'fileNames'))
def get_allMessages(isCompleted, fileNames):
    if isCompleted:
        allMessages = parse_data(fileNames[0])
        if isinstance(allMessages, pd.DataFrame):
            allMessages['MessageId'] = allMessages['UniversalMessageId']
            allMessages = allMessages.set_index('UniversalMessageId')
            allMessages_json = allMessages.to_json()

            upload_status = ' Finished uploading ：' + fileNames[0]
            return allMessages_json, upload_status
        else:
            upload_status = fileNames[0] + ' cannot be uploaded. Its file format is not supported! Please upload another file (csv, txt, tsv, xls).'
            return None,  upload_status
    else:
        return None, dash.no_update

# output message 1
@app.callback(
    Output('feature_slider_output', 'children'),
    Input('words_freq_threshold', 'children'),
    Input('top_words_num', 'value'),
    Input('feature_words_list', 'children'))
def update_output(words_freq_threshold, top_words_num, feature_words_list):
    return 'Common words that appear in >= {} clusters have been removed from the list of top {} stemmed words.            \n The remaining words are used as the feature words: {}'.format(words_freq_threshold, top_words_num, feature_words_list)

### Analysis
## Section 1
# Document-word matrix for all tweets
@app.callback(Output('docWordMatrix_orig_json', 'children'),
    [Input('allMessages_json', 'children')])
def make_full_docWordMatrix(allMessages_json):
    if allMessages_json is not None:
        # de-json-ify cleaned tweets
        allMessages = pd.read_json(allMessages_json)
        if 'cleaned' not in allMessages.columns:
            allMessages['cleaned'] = [preProcessingFcn(tweet) for tweet in allMessages.Message.str.lower().values.tolist()]
        cleanedTweets = allMessages['cleaned']
        # create document-word matrix
        vectorizer = CountVectorizer(strip_accents='unicode', min_df=5, binary=False)
        docWordMatrix_orig = vectorizer.fit_transform(cleanedTweets)
        docWordMatrix_orig = docWordMatrix_orig.astype(dtype='float64')
        # save as sparse document-word matrix as json file
        rows_orig, cols_orig = docWordMatrix_orig.nonzero()
        data_orig = docWordMatrix_orig.data
        docWordMatrix_orig_json = json.dumps({'rows_orig':rows_orig.tolist(), 'cols_orig':cols_orig.tolist(),
            'data_orig':data_orig.tolist(), 'dims_orig':[docWordMatrix_orig.shape[0], docWordMatrix_orig.shape[1]],
            'feature_names':vectorizer.get_feature_names(), 'indices':allMessages.index.tolist()})
        return docWordMatrix_orig_json

# restrict document-word matrix based on keep and remove words
# restrict document-word matrix based on keep and remove words
@app.callback(Output('docWordMatrix_json', 'children'),
    Output('subset_info_remove', 'children'),
    Output('subset_info_keep', 'children'),
    Output('new_indices_json', 'children'),
    Input('button', 'n_clicks'),
    Input('docWordMatrix_orig_json', 'children'),
    dash.dependencies.State('removeWords', 'value'),
    dash.dependencies.State('keepWords', 'value'))
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
            	# get individual words
                remove_words = removeWords.split(',')
                # pre-process removal words (stemming, etc.)
                remove_words_cleaned = [preProcessingFcn(word) for word in remove_words]
                # get columns of word-document matrix associated with remove words
                remove_words_ori_final = []
                remove_words_stemmed_final = []
                incorrect_words_ori = []
                remove_cols = []
                i = 0
                for word in remove_words_cleaned:
                    if word in feature_names:
                        remove_words_ori_final.append(remove_words[i])
                        remove_words_stemmed_final.append(word)
                        remove_cols.append(feature_names.index(word))
                    else:
                        incorrect_words_ori.append(remove_words[i])
                    i = i + 1
                # remove_cols = [feature_names.index(word) for word in remove_words_cleaned if word in feature_names]
                # restrict document-word matrix to columns of remove words and take the sum of the rows
                remove_rowSums = docWordMatrix_update.tocsr()[:,remove_cols].sum(axis=1)
                # add remove words to print statement
                subset_info_remove1 = ''
                if len(remove_words_ori_final) > 0:
                    subset_info_remove1 = 'Removing tweet(s) that contain: '
                    for i in range(len(remove_words_ori_final)):
                        subset_info_remove1 += remove_words_ori_final[i] + '(' + remove_words_stemmed_final[i] + '),'
                    subset_info_remove1 = subset_info_remove1[:-1] + '. '

                subset_info_remove2 = ''
                if len(incorrect_words_ori) > 0:
                    subset_info_remove2 = 'Input word(s) not valid: '
                    for i in range(len(incorrect_words_ori)):
                        subset_info_remove2 += incorrect_words_ori[i] + ','
                    subset_info_remove2 = subset_info_remove2[:-1] + '. '

                subset_info_remove = subset_info_remove1 + subset_info_remove2
        if keepWords is not None: # if any keeps words entered
            if len(keepWords)>0:
            	# get individual words
                keep_words = keepWords.split(',')
                # pre-process keeping words
                keep_words_cleaned = [preProcessingFcn(word) for word in keep_words]
                # get columns of word-document matrix associated with keep words
                keep_words_ori_final = []
                keep_words_stemmed_final = []
                incorrect_words_ori = []
                keep_cols = []
                i = 0
                for word in keep_words_cleaned:
                    if word in feature_names:
                        keep_words_ori_final.append(keep_words[i])
                        keep_words_stemmed_final.append(word)
                        keep_cols.append(feature_names.index(word))
                    else:
                        incorrect_words_ori.append(keep_words[i])
                    i = i + 1
                # keep_cols = [feature_names.index(word) for word in keep_words_cleaned if word in feature_names]
                # restrict document-word matrix to columns of keep words and take sum of the rows
                keep_rowSums = docWordMatrix_update.tocsr()[:,keep_cols].sum(axis=1)
                # add keep words to print statement
                subset_info_keep1 = ''
                if len(keep_words_ori_final) > 0:
                    subset_info_keep1 = 'Keeping tweet(s) that contain: '
                    for i in range(len(keep_words_ori_final)):
                        subset_info_keep1 += keep_words_ori_final[i] + '(' + keep_words_stemmed_final[i] + '),'
                    subset_info_keep1 = subset_info_keep1[:-1] + '. '

                subset_info_keep2 = ''
                if len(incorrect_words_ori) > 0:
                    subset_info_keep2 = 'Input word(s) not valid: '
                    for i in range(len(incorrect_words_ori)):
                        subset_info_keep2 += incorrect_words_ori[i] + ','
                    subset_info_keep2 = subset_info_keep2[:-1] + '. '

                subset_info_keep = subset_info_keep1 + subset_info_keep2

        # restrict to remove/keep words
        # new doc-word matrix (and indices) as rows where total keep words >0 and total remove words =0
        new_rows = [i for i in range(len(remove_rowSums)) if remove_rowSums[i]==0 and keep_rowSums[i]>=1]
        num_new_rows = len(new_rows)
        if num_new_rows == 0:
            new_rows = [i for i in range(len(remove_rowSums))]
            num_new_rows = len(new_rows)
            subset_info_remove = str(num_new_rows) + ' tweets in total. '; subset_info_keep = 'Meaningless! Remove and keep the same word(s). Please re-enter the words.'
        else:
            subset_info_keep = subset_info_keep + ' ' + str(num_new_rows) + ' tweets in total. '
        
        docWordMatrix_update = docWordMatrix_update.tocsr()[new_rows]

        # perform tf-idf on word-document matrix
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


# display options based on default or advanced options
@app.callback(Output('advanced_options', 'style'),
    Input('default_or_advanced', 'value'))
def adv_options_formatting(default_or_advanced):
    # if choosing advanced option, display advanced options (i.e., block display)
    # otherwise if default option, do not display advanced options (i.e., hidden display)
    if default_or_advanced=='default':
        style_advanced = {'display': 'none'}
    else:
        style_advanced = {'display': 'block'}
    return style_advanced

# Do dimension reduction, clustering
# create plot and table to output
@app.callback(
    Output('dimRed_cluster_plot', 'figure'),
    Output('dimRed_cluster_plot2', 'figure'),
    Output('dimRed_cluster_plot3', 'figure'),
    Output('clusters_table', 'columns'),
    Output('clusters_table', 'data'),
    Output('unique_cluster_options', 'options'),
    Output('download_tweet_cluster', 'options'),
    Output('cluster_ids_json', 'children'),
    Output('feature_cluster_dot_plot', 'figure'),
    Output('words_freq_threshold', 'children'),
    Output('feature_words_list', 'children'),
    Input('docWordMatrix_orig_json', 'children'),
    Input('docWordMatrix_json', 'children'),
    Input('allMessages_json', 'children'),
    Input('new_indices_json', 'children'),
    Input('dimRed1_method', 'value'),
    Input('dimRed1_dims', 'value'),
    Input('dimRed2_method', 'value'),
    Input('clustering_when', 'value'),
    Input('clustering_method1', 'value'),
    Input('num_clusters', 'value'),
    Input('min_obs', 'value'),
    Input('top_words_num','value'),
    Input('feature_perc_threshold','value'))
def dimRed_and_clustering(docWordMatrix_orig_json, docWordMatrix_json, allMessages_json, new_indices_json,
    dimRed1_method, dimRed1_dims, dimRed2_method, 
    clustering_when, clustering_method1, num_clusters, min_obs, top_words_num, feature_perc_threshold):
    if docWordMatrix_json is None:
        display_df = pd.DataFrame(data={'Cluster': [], 'Proportion of Tweets':[],
            'Number of Tweets':[], 'Top 5 Stemmed Words':[]})
        display_df_columns = [{'name': col, 'id': col} for col in display_df.columns]
        display_df_data = display_df.to_dict(orient='records')
        unique_cluster_options = [{'label': '', 'value': ''}]
        return {}, {}, {}, display_df_columns, display_df_data, unique_cluster_options, unique_cluster_options, None, {}, None, None
    # read in document-word matrix
    json_data = json.loads(docWordMatrix_json)
    data = json_data['data_new']
    rows = json_data['rows_new']
    cols = json_data['cols_new']
    dims = json_data['dims_new']
    feature_names = json_data['feature_names']        
    json_indices = json.loads(new_indices_json)
    indices = json_indices['indices']
    docWordMatrix = csc_matrix((data, (rows, cols)), shape=(dims[0], dims[1]))
    # read in allMessages
    allMessages = pd.read_json(allMessages_json)
    # do stage 1 dimension reduction
    if dimRed1_method == 'pca':
        dimRed1 = dimred_PCA(docWordMatrix, ndims=dimRed1_dims)
    elif dimRed1_method == 'umap':
        dimRed1 = dimred_UMAP(docWordMatrix, ndims=dimRed1_dims)
    # do stage 2 dimension reduction (if any)
    if dimRed1_dims > 2:
        if dimRed2_method == 'pca':
            dimRed2 = dimred_PCA(dimRed1, ndims=2)
        elif dimRed2_method == 'umap':
            dimRed2 = dimred_UMAP(dimRed1, ndims=2)
    else:
        dimRed2 = dimRed1
    # Clustering
    # get matrix at proper stage
    if clustering_when == 'before_stage1':
        clustering_data = docWordMatrix
    elif clustering_when == 'btwn':
        clustering_data = dimRed1
    elif clustering_when == 'after_stage2':
        clustering_data = dimRed2
    # perform clustering
    if clustering_method1 == 'gmm':
        clusters = cluster_gmm(clustering_data, num_clusters=num_clusters)
    elif clustering_method1 == 'k-means':
        clusters = cluster_kmeans(clustering_data, num_clusters=num_clusters)
    elif clustering_method1 == 'hdbscan':
        clusters = cluster_hdbscan(clustering_data, min_obs=min_obs)
    # make plot
    allMessages_plot = allMessages.loc[indices]
    allMessages_plot['Cluster'] = clusters # color by cluster
    allMessages_plot['Text'] = allMessages_plot['Message'].apply(lambda t: "<br>".join(textwrap.wrap(t))) # make tweet text display cleanly
    allMessages_plot['coord1'] = dimRed2[:,0] # x-coordinate
    allMessages_plot['coord2'] = dimRed2[:,1] # y-coordinate
    dimRed_cluster_plot = px.scatter(allMessages_plot, x='coord1', y='coord2', color='Cluster',
        hover_data=['Text', 'MessageId'])
    dimRed_cluster_plot.update_layout(clickmode='event+select')

    # representative words and tweet IDs for each cluster
    # read in document word matrix
    # json_data = json.loads(docWordMatrix_orig_json)
    # data = json_data['data_orig']
    # rows = json_data['rows_orig']
    # cols = json_data['cols_orig']
    # dims = json_data['dims_orig']
    # feature_names = json_data['feature_names']
    # indices = json_data['indices']
    # docWordMatrix = csc_matrix((data, (rows, cols)), shape=(dims[0], dims[1]))
    # table data
    clusters_data = []; topWords_data = []; prop_data = []; num_data = []; num_unique = []
    cluster_ids_dict = dict()
    for i in set(allMessages_plot['Cluster']):
        # restrict docWordMatrix to rows in cluster i
        clusteri = docWordMatrix[allMessages_plot['Cluster']==i, :]
        # get most common words
        colSumsi = np.squeeze(np.asarray(clusteri.sum(axis=0)))
        topN = np.sort(colSumsi)[-top_words_num]
        topWordsi = [feature_names[j] for j in range(len(colSumsi)) if colSumsi[j]>=topN][0:top_words_num]
        # add to display data
        clusters_data.append(i)
        topWords_data.append(" ".join(topWordsi))
        prop_data.append(round(clusteri.shape[0]/docWordMatrix.shape[0], 3))
        num_data.append(clusteri.shape[0])
        subset_data = allMessages_plot[allMessages_plot['Cluster']==i]
        num_unique.append(len(pd.unique(subset_data['Message'])))
        cluster_ids_dict[i] = allMessages_plot[allMessages_plot['Cluster']==i].index.tolist()
    topWords_str = 'Top ' + str(top_words_num) + ' Stemmed Words'
    display_df = pd.DataFrame(data={'Cluster':[int(x) for x in clusters_data], 'Proportion of Tweets':prop_data,
        'Number of Tweets':num_data, 'Number of Unique Tweets': num_unique, topWords_str:topWords_data})
    display_df_columns = [{'name': col, 'id': col} for col in display_df.columns]
    display_df_data = display_df.to_dict(orient='records')
    cluster_ids_dict['all'] = indices

    unique_cluster_options = [{'label':i, 'value':i} for i in clusters_data] + [{'label':'all', 'value':'all'}]
    
    # make dot plot to visualize feature words
    words_freq_threshold, feature_words = choose_features(display_df, top_words_num = top_words_num, threshold = feature_perc_threshold)
    temp_cluster_data = allMessages_plot[['Cluster']]
    group_idx = list(temp_cluster_data["Cluster"])
    DT_mtx  = pd.DataFrame.sparse.from_spmatrix(docWordMatrix).to_numpy()
    
    feature_words_table = feature_table(feature_names, feature_words, group_idx, DT_mtx)
    
    feature_cluster_dot_plot = px.scatter(feature_words_table, #dataframe
                                     x="Group", #x
                                     y="Word", #y
                                     size="percentage", #bubble size
                                     color="mean",#bubble color
                                     color_continuous_scale='viridis', #color theme
                                     title="Feature words composition", #chart
                                     height=1000)
    feature_cluster_dot_plot.update_layout(xaxis_tickangle=30,#angle of the tick on x-axis
                                       title=dict(x=0.5), #set the title in center
                                       xaxis_tickfont=dict(size=12), #set the font for x-axis
                                       yaxis_tickfont=dict(size=12), #set the font for y-axis
                                       margin=dict(l=100, r=20, t=50, b=20), #set the margin
                                       # paper_bgcolor="LightSteelblue", #set the background color for chart
                                       clickmode='event+select')
    
    return dimRed_cluster_plot, dimRed_cluster_plot, dimRed_cluster_plot, display_df_columns, display_df_data, unique_cluster_options, unique_cluster_options, json.dumps(cluster_ids_dict), feature_cluster_dot_plot, words_freq_threshold, feature_words

# Displays for clustering options
# Input as number of clusters for k-means, gmm
# Input as minimum observations for hdbscan
@app.callback(Output('num_clustering_input_container', 'style'),
    Output('min_size_input_container', 'style'),
    Input('clustering_method1', 'value'))
def cluster_options_formatting(clustering_method):
    if clustering_method in ['gmm', 'k-means']:
        style_nclusters={'display':'block'}
        style_minObs={'display':'none'}
    else:
        style_nclusters={'display':'none'}
        if clustering_method in ['hdbscan']:
            style_minObs={'display':'block'}
    return style_nclusters, style_minObs


## Section 2: Characteristics of clusters
# display options based on cluster or manual options
# Displays for clustering options
# Input as number of clusters for k-means, gmm
# Input as minimum observations for hdbscan
@app.callback(Output('cluster_select_options', 'style'),
    Output('manual_select_options', 'style'),
    Input('cluster_or_manual', 'value'))
def tweets_select_options_formatting(selecting_method):
    if selecting_method == 'cluster':
        style_cluster={'display':'block'}
        style_manual={'display':'none'}
    else:
        style_cluster={'display':'none'}
        if selecting_method == 'manual':
            style_manual={'display':'block'}
    return style_cluster, style_manual

'''
@app.callback(Output('manual_select_options', 'style'),
    Input('cluster_or_manual', 'value'))
def manual_options_formatting(cluster_or_manual):
    # if choosing manual option, display manual options (i.e., block display)
    # otherwise if default option, do not display manual options (i.e., hidden display)
    if default_or_advanced=='cluster':
        style_advanced = {'display': 'none'}
    else:
        style_advanced = {'display': 'block'}
    return style_advanced
'''

@ app.callback([Output('randomSample_title', 'children'),
    Output('table_randomSample', 'children'),
    Output('randomSampleFollow_title', 'children'),
    Output('table_randomSampleFollow', 'children'),
    Output('randomSampleUnique_title', 'children'),
    Output('table_randomSampleUnique', 'children'),
    Output('freq_raw_plot', 'figure'),
    Output('freq_raw_plot_container', 'style'),
    Output('freq_norm_plot', 'figure'),
    Output('freq_norm_plot_container', 'style'),
    Output('plot_geo', 'figure'),
    Output('geo_plot_container', 'style'),
    Output('plot_geo_norm', 'figure'),
    Output('geo_plot_norm_container', 'style')],
    [Input('unique_cluster_options', 'value'),
    Input('cluster_chars', 'value'),
    Input('cluster_ids_json', 'children'),
    Input('allMessages_json', 'children')])
def tweet_set_characteristics(unique_cluster_options, cluster_chars, cluster_ids_json, allMessages_json):
    ndisplay = 5 # number of tweets to display for each random sample
    randomSample_title = ''
    cluster_randomSample = pd.DataFrame()
    randomSampleFollow_title = ''
    cluster_randomSampleFollow = pd.DataFrame()
    randomSampleUnique_title = ''
    cluster_randomSampleUnique = pd.DataFrame()
    cluster_rawFreq = pd.DataFrame()
    freq_raw_plot = {}
    freq_raw_plot_container = {'display': 'none'}
    freq_norm_plot = {}
    freq_norm_plot_container = {'display': 'none'}
    geo_data = pd.DataFrame()
    geo_plot = {}
    geo_plot_container = {'display': 'none'}
    geo_norm_plot = {}
    geo_plot_norm_container = {'display': 'none'}
    # make data frames
    if unique_cluster_options is not None and cluster_chars is not None:
        # get allMessages
        allMessages = pd.read_json(allMessages_json)
        # get tweet ids for tweets in each cluster
        json_ids = json.loads(cluster_ids_json)
        for clust in unique_cluster_options:
            # get tweet IDs for tweets in the specific cluster
            cluster_ids = json_ids[clust]
            # restrict allMessages to clust
            cluster_subset = allMessages.loc[cluster_ids,:]
            # random sample of tweets for each cluster
            if 'randSamp' in cluster_chars:
                randomSample_title = 'Random Sample of Tweets:'
                # make a table with each column a cluster
                # get random sample
                cluster_sample = cluster_subset['Message'].sample(n=min(ndisplay, cluster_subset.shape[0]))
                while len(cluster_sample) < ndisplay:
                    cluster_sample = cluster_sample.append(pd.Series(['']))
                # attach to data frame as new column
                cluster_randomSample.reset_index(drop=True, inplace=True)
                cluster_sample.reset_index(drop=True, inplace=True)
                cluster_randomSample[clust] = cluster_sample
            # random sample weighted by number of followers
            if 'randSamp_follow' in cluster_chars:
                randomSampleFollow_title = 'Random Sample of Tweets, Weighted by Follower Count:'
                cluster_sampleFollow = cluster_subset['Message'].sample(
                    n=min(ndisplay, cluster_subset.shape[0]),
                    weights=[max(0, weight) for weight in cluster_subset['Sender Followers Count']])
                while len(cluster_sampleFollow) < ndisplay:
                    cluster_sampleFollow = cluster_sampleFollow.append(pd.Series(['']))
                # attach to data frame as new column
                cluster_randomSampleFollow.reset_index(drop=True, inplace=True)
                cluster_sampleFollow.reset_index(drop=True, inplace=True)
                cluster_randomSampleFollow[clust] = cluster_sampleFollow
            if 'randSamp_unique' in cluster_chars:
                randomSampleUnique_title = 'Randon Sample of Unique Tweets:'
                uniqueTweets = cluster_subset['Message'].drop_duplicates()
                cluster_sampleUnique = uniqueTweets.sample(n=min(ndisplay, uniqueTweets.shape[0]))
                while len(cluster_sampleUnique) < ndisplay:
                    cluster_sampleUnique = cluster_sampleUnique.append(pd.Series(['']))
                # attach to data frame as new column
                cluster_randomSampleUnique.reset_index(drop=True, inplace=True)
                cluster_sampleUnique.reset_index(drop=True, inplace=True)
                cluster_randomSampleUnique[clust] = cluster_sampleUnique
            # frequency: raw number of tweets per day (use this later to get proportion)
            if 'freq_raw' in cluster_chars or 'freq_norm' in cluster_chars:
                cluster_freqCounts = cluster_subset['Day'].value_counts()
                cluster_rawFreq = cluster_rawFreq.join(cluster_freqCounts, how='outer')
                cluster_rawFreq = cluster_rawFreq.rename(columns={'Day': clust})
            # state map of frequency
            if 'geo' in cluster_chars or 'geo_norm' in cluster_chars:
                cluster_stateCounts = cluster_subset['State'].value_counts().astype('float64')
                geo_data = geo_data.join(cluster_stateCounts, how='outer')
                geo_data = geo_data.rename(columns={'State': clust})
        # make plots
        if 'freq_raw' in cluster_chars or 'freq_norm' in cluster_chars:
            cluster_rawFreq = cluster_rawFreq.fillna(0)
            cluster_rawFreq = cluster_rawFreq.sort_index()
            if 'freq_raw' in cluster_chars:
                freq_raw_plot = px.line(cluster_rawFreq, x=cluster_rawFreq.index, y=cluster_rawFreq.columns,
                    title='Frequency of Tweets Over Time')
                freq_raw_plot.update_xaxes(title_text='Date'); freq_raw_plot.update_yaxes(title_text='Frequency')
                freq_raw_plot_container = {'display': 'block'}
            if 'freq_norm' in cluster_chars:
                cluster_normFreq = cluster_rawFreq.div(cluster_rawFreq.sum(axis=0))
                freq_norm_plot = px.line(cluster_normFreq, x=cluster_normFreq.index, y=cluster_normFreq.columns,
                    title='Proportion of Tweets Over Time')
                freq_norm_plot.update_xaxes(title_text='Date'); freq_norm_plot.update_yaxes(title_text='Proportion')
                freq_norm_plot_container = {'display': 'block'}
        if 'geo' in cluster_chars:
            geo_plot_container = {'display': 'block'}
            # set up state-level frequency subplot
            cols = 2
            rows = math.ceil(len(unique_cluster_options)/2)
            geo_plot = make_subplots(
                rows = rows, cols=cols, 
                specs = [[{'type': 'choropleth'} for c in np.arange(cols)] for r in np.arange(rows)],
                subplot_titles = list(unique_cluster_options)
            )
            i=0
            for clust in unique_cluster_options:
                geo_plot.add_trace(go.Choropleth(
                    locations = [us_state_abbrev[name] for name in geo_data.index],
                    z = geo_data[clust],
                    locationmode = 'USA-states', marker_line_color='white',
                    zmin=0, zmax= geo_data.max(axis=0, skipna=True).max(skipna=True),
                    colorbar_title='Frequency'
                ), row=i//cols+1, col=i%cols+1)
                i += 1
            geo_plot.update_layout(
                title_text = 'Frequency of Tweets by State',
                **{'geo' + str(i) + '_scope': 'usa' for i in ['']+np.arange(2,rows*cols+1).tolist()}
            )
        if 'geo_norm' in cluster_chars:
            geo_plot_norm_container = {'display': 'block'}
            # divide raw number of tweets by state population
            # geo_data = pd.to_numeric(geo_data.iloc[:, 0], downcast="float").to_frame()
            geo_perCapita = geo_data.divide(state_pops, axis=0)
            # set up state-level 'tweets-per-capita' subplot
            cols = 2
            rows = math.ceil(len(unique_cluster_options)/2)
            geo_norm_plot = make_subplots(
                rows = rows, cols=cols, 
                specs = [[{'type': 'choropleth'} for c in np.arange(cols)] for r in np.arange(rows)],
                subplot_titles = list(unique_cluster_options)
            )
            i=0
            for clust in unique_cluster_options:
                geo_norm_plot.add_trace(go.Choropleth(
                    locations = [us_state_abbrev[name] for name in geo_perCapita.index],
                    z = geo_perCapita[clust],
                    locationmode = 'USA-states', marker_line_color='white',
                    zmin=0, zmax= geo_perCapita.max(axis=0, skipna=True).max(skipna=True),
                    colorbar_title='Frequency'
                ), row=i//cols+1, col=i%cols+1)
                i += 1
            geo_norm_plot.update_layout(
                title_text = 'Frequency of Tweets by State, Normalized By State Population',
                **{'geo' + str(i) + '_scope': 'usa' for i in ['']+np.arange(2,rows*cols+1).tolist()}
            )

    return randomSample_title, generate_table(cluster_randomSample), randomSampleFollow_title, generate_table(cluster_randomSampleFollow), randomSampleUnique_title, generate_table(cluster_randomSampleUnique), freq_raw_plot, freq_raw_plot_container, freq_norm_plot, freq_norm_plot_container, geo_plot, geo_plot_container, geo_norm_plot, geo_plot_norm_container

# options to download tweets
@app.callback(Output('download', 'data'),
    Input('download_button', 'n_clicks'),
    Input('cluster_ids_json', 'children'),
    Input('allMessages_json', 'children'),
    dash.dependencies.State('download_tweet_cluster', 'value'),
    prevent_initial_call=True)
def generate_csv(n_clicks, cluster_ids_json, allMessages_json, download_tweet_cluster):
    # subset to tweets from selected cluster
    if download_tweet_cluster is not None:
        allMessages = pd.read_json(allMessages_json)
        json_ids = json.loads(cluster_ids_json)
        cluster_ids = json_ids[download_tweet_cluster]
        cluster_subset = allMessages.loc[cluster_ids,:]
        filename = 'cluster_' + str(download_tweet_cluster) + '.csv'
        # return csv
        return send_data_frame(cluster_subset.to_csv, filename=filename)
    else:
        pass


## Section test: selected tweets

@app.callback(
        Output('output_tweet_count', 'children'),
        Output('selected_subset_json', 'children'),
        Input('dimRed_cluster_plot3','selectedData'),
        Input('allMessages_json', 'children'))
def generate_selected_table(selectData, allMessages_json):
    if selectData is not None:
        # identified selected tweets' IDs
        points_info = selectData['points']
        ID_list = []
        for point in points_info:
            ID_list.append(point['customdata'][1])
        
        output_tweet_txt = 'You have selected {} tweets.'.format(str(len(ID_list)))

        allMessages = pd.read_json(allMessages_json)
        selected_subset = allMessages.loc[ID_list,:]
        selected_subset_json = selected_subset.to_json()
        # return table
        return output_tweet_txt, selected_subset_json
    else:
        return None, None

@app.callback(
        Output('download1', 'data'),
        Input('download_button1', 'n_clicks'),
        dash.dependencies.State('selected_subset_json', 'children'),
        dash.dependencies.State('selected_csv_name', 'value'),
        prevent_initial_call=True)
def generate_selected_csv(n_clicks, selected_subset_json, selected_csv_name):
    if selected_subset_json is not None:
        if selected_csv_name is not None:
            filename = 'selected_tweets_' + str(selected_csv_name) + '.csv'
        else:
            filename = 'selected_tweets_' + str(n_clicks) + '.csv'
        selected_subset = pd.read_json(selected_subset_json)

        # return csv
        return send_data_frame(selected_subset.to_csv, filename=filename)
    else:
        pass
    
# characteristics of selected tweets
@ app.callback([Output('selected_randomSample_title', 'children'),
    Output('selected_table_randomSample', 'children'),
    Output('selected_randomSampleFollow_title', 'children'),
    Output('selected_table_randomSampleFollow', 'children'),
    Output('selected_randomSampleUnique_title', 'children'),
    Output('selected_table_randomSampleUnique', 'children'),
    Output('selected_freq_raw_plot', 'figure'),
    Output('selected_freq_raw_plot_container', 'style'),
    Output('selected_freq_norm_plot', 'figure'),
    Output('selected_freq_norm_plot_container', 'style'),
    Output('selected_plot_geo', 'figure'),
    Output('selected_geo_plot_container', 'style'),
    Output('selected_plot_geo_norm', 'figure'),
    Output('selected_geo_plot_norm_container', 'style')],
    [Input('selected_chars', 'value'),
    Input('selected_subset_json', 'children')])
def tweet_set_characteristics(selected_chars, selected_subset_json):
    ndisplay = 5 # number of tweets to display for selected tweets
    randomSample_title = ''
    selected_randomSample = pd.DataFrame()
    randomSampleFollow_title = ''
    selected_randomSampleFollow = pd.DataFrame()
    randomSampleUnique_title = ''
    selected_randomSampleUnique = pd.DataFrame()
    selected_rawFreq = pd.DataFrame()
    freq_raw_plot = {}
    freq_raw_plot_container = {'display': 'none'}
    freq_norm_plot = {}
    freq_norm_plot_container = {'display': 'none'}
    geo_data = pd.DataFrame()
    geo_plot = {}
    geo_plot_container = {'display': 'none'}
    geo_norm_plot = {}
    geo_plot_norm_container = {'display': 'none'}
    # make data frames
    if selected_chars is not None:
        # get allMessages
        selected_subset = pd.read_json(selected_subset_json)
        # random sample of tweets for the selected set of tweets 
        if 'randSamp' in selected_chars:
            randomSample_title = 'Random Sample of Tweets:'
            # make a table with each column a cluster
            # get random sample
            selected_sample = selected_subset['Message'].sample(n=min(ndisplay, selected_subset.shape[0]))
            while len(selected_sample) < ndisplay:
                selected_sample = selected_sample.append(pd.Series(['']))
            # attach to data frame as new column
            selected_randomSample.reset_index(drop=True, inplace=True)
            selected_sample.reset_index(drop=True, inplace=True)
            selected_randomSample["selected"] = selected_sample
            # random sample weighted by number of followers
        if 'randSamp_follow' in selected_chars:
            randomSampleFollow_title = 'Random Sample of Tweets, Weighted by Follower Count:'
            selected_sampleFollow = selected_subset['Message'].sample(
                n=min(ndisplay, selected_subset.shape[0]),
                weights=[max(0, weight) for weight in selected_subset['Sender Followers Count']])
            while len(selected_sampleFollow) < ndisplay:
                selected_sampleFollow = selected_sampleFollow.append(pd.Series(['']))
            # attach to data frame as new column
            selected_randomSampleFollow.reset_index(drop=True, inplace=True)
            selected_sampleFollow.reset_index(drop=True, inplace=True)
            selected_randomSampleFollow["selected"] = selected_sampleFollow
        if 'randSamp_unique' in selected_chars:
            randomSampleUnique_title = 'Randon Sample of Unique Tweets:'
            uniqueTweets = selected_subset['Message'].drop_duplicates()
            selected_sampleUnique = uniqueTweets.sample(n=min(ndisplay, uniqueTweets.shape[0]))
            while len(selected_sampleUnique) < ndisplay:
                selected_sampleUnique = selected_sampleUnique.append(pd.Series(['']))
            # attach to data frame as new column
            selected_randomSampleUnique.reset_index(drop=True, inplace=True)
            selected_sampleUnique.reset_index(drop=True, inplace=True)
            selected_randomSampleUnique["selected"] = selected_sampleUnique
        # frequency: raw number of tweets per day (use this later to get proportion)
        if 'freq_raw' in selected_chars or 'freq_norm' in selected_chars:
            selected_freqCounts = selected_subset['Day'].value_counts()
            selected_rawFreq = selected_rawFreq.join(selected_freqCounts, how='outer')
            selected_rawFreq = selected_rawFreq.rename(columns={'Day': "selected"})
        # state map of frequency
        if 'geo' in selected_chars or 'geo_norm' in selected_chars:
            selected_stateCounts = selected_subset['State'].value_counts().astype('float64')
            geo_data = geo_data.join(selected_stateCounts, how='outer')
            geo_data = geo_data.rename(columns={'State': "selected"})
        # make plots
        if 'freq_raw' in selected_chars or 'freq_norm' in selected_chars:
            selected_rawFreq = selected_rawFreq.fillna(0)
            selected_rawFreq = selected_rawFreq.sort_index()
            if 'freq_raw' in selected_chars:
                freq_raw_plot = px.line(selected_rawFreq, x=selected_rawFreq.index, y=selected_rawFreq.columns,
                                        title='Frequency of Tweets Over Time')
                freq_raw_plot.update_xaxes(title_text='Date'); freq_raw_plot.update_yaxes(title_text='Frequency')
                freq_raw_plot_container = {'display': 'block'}
            if 'freq_norm' in selected_chars:
                selected_normFreq = selected_rawFreq.div(selected_rawFreq.sum(axis=0))
                freq_norm_plot = px.line(selected_normFreq, x=selected_normFreq.index, y=selected_normFreq.columns,
                                         title='Proportion of Tweets Over Time')
                freq_norm_plot.update_xaxes(title_text='Date'); freq_norm_plot.update_yaxes(title_text='Proportion')
                freq_norm_plot_container = {'display': 'block'}
        if 'geo' in selected_chars:
            geo_plot_container = {'display': 'block'}
            # set up state-level frequency subplot
            cols = 2
            rows = 1
            geo_plot = make_subplots(
                rows = rows, cols=cols, 
                specs = [[{'type': 'choropleth'} for c in np.arange(cols)] for r in np.arange(rows)],
                subplot_titles = ["selected"]
            )
            i=0
        
            geo_plot.add_trace(go.Choropleth(
                locations = [us_state_abbrev[name] for name in geo_data.index],
                z = geo_data["selected"],
                locationmode = 'USA-states', marker_line_color='white',
                zmin=0, zmax= geo_data.max(axis=0, skipna=True).max(skipna=True),
                colorbar_title='Frequency'
            ), row=i//cols+1, col=i%cols+1)

            geo_plot.update_layout(
                title_text = 'Frequency of Tweets by State',
                **{'geo' + str(i) + '_scope': 'usa' for i in ['']+np.arange(2,rows*cols+1).tolist()}
            )
        if 'geo_norm' in selected_chars:
            geo_plot_norm_container = {'display': 'block'}
            # divide raw number of tweets by state population
            # geo_data = pd.to_numeric(geo_data.iloc[:, 0], downcast="float").to_frame()
            geo_perCapita = geo_data.divide(state_pops, axis=0)
            # set up state-level 'tweets-per-capita' subplot
            cols = 2
            rows = 1
            geo_norm_plot = make_subplots(
                rows = rows, cols=cols, 
                specs = [[{'type': 'choropleth'} for c in np.arange(cols)] for r in np.arange(rows)],
                subplot_titles = ["selected"]
            )
            i=0
            geo_norm_plot.add_trace(go.Choropleth(
                locations = [us_state_abbrev[name] for name in geo_perCapita.index],
                z = geo_perCapita["selected"],
                locationmode = 'USA-states', marker_line_color='white',
                zmin=0, zmax= geo_perCapita.max(axis=0, skipna=True).max(skipna=True),
                colorbar_title='Frequency'
            ), row=i//cols+1, col=i%cols+1)
            
            geo_norm_plot.update_layout(
                title_text = 'Frequency of Tweets by State, Normalized By State Population',
                **{'geo' + str(i) + '_scope': 'usa' for i in ['']+np.arange(2,rows*cols+1).tolist()}
            )

    return randomSample_title, generate_table(selected_randomSample), randomSampleFollow_title, generate_table(selected_randomSampleFollow), randomSampleUnique_title, generate_table(selected_randomSampleUnique), freq_raw_plot, freq_raw_plot_container, freq_norm_plot, freq_norm_plot_container, geo_plot, geo_plot_container, geo_norm_plot, geo_plot_norm_container

## Section 3: Individual Tweet
@app.callback(Output('click_data', 'children'),
	Input('dimRed_cluster_plot2', 'clickData'))
def display_click_data(clickData):
	return json.dumps(clickData)

## function to get the current directory
def get_file_directory(file):
    return os.path.dirname(os.path.abspath(file))

if __name__=='__main__':
    app.run_server(host='0.0.0.0', debug=True)

    # clear the "temp" folder after 7 days
    now = time.time()
    cutoff = now - (7 * 86400)

    files = os.listdir(os.path.join(get_file_directory(__file__), "temp"))
    file_path = os.path.join(get_file_directory(__file__), "temp/")
    for xfile in files:
        if os.path.isfile(str(file_path) + xfile):
            t = os.stat(str(file_path) + xfile)
            c = t.st_ctime

            # delete file if older than 10 days
            if c < cutoff:
                os.remove(str(file_path) + xfile)
