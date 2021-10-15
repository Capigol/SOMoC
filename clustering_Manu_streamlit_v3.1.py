#!/usr/bin/env python
# coding: utf-8

# In[1]:
    
###### CLUSTERING MANU ##########
# Needed packages

import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from validclust import dunn
from rdkit import Chem
import umap
import janitor
from janitor import chemistry
import streamlit as st
from pathlib import Path
import base64
from PIL import Image
from datetime import date
import random
from statistics import mean, stdev
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#%%

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='LIDEB Tools - SOMoC',
    layout='wide')

######
# Function to put a picture as header   
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


image = Image.open('cropped-header.png')
st.image(image)

st.write("&nbsp[![Website](https://img.shields.io/badge/website-LIDeB-blue)](https://lideb.biol.unlp.edu.ar)&nbsp[![Twitter Follow](https://img.shields.io/twitter/follow/LIDeB_UNLP?style=social)](https://twitter.com/intent/follow?screen_name=LIDeB_UNLP)")
st.subheader(":pushpin:" "About Us")
st.markdown("We are a drug discovery team with an interest in the development of publicly available open-source customizable cheminformatics tools to be used in computer-assisted drug discovery. We belong to the Laboratory of Bioactive Research and Development (LIDeB) of the National University of La Plata (UNLP), Argentina. Our research group is focused on computer-guided drug repurposing and rational discovery of new drug candidates to treat epilepsy and neglected tropical diseases.")



# Introduction
#---------------------------------#

st.write("""
# LIDeB Tools - SOMoC

**It is a free web-application to cluster molecules based in molecular fingerprints**

The tool uses the following packages [RDKIT](https://www.rdkit.org/docs/index.html), [UMAP](https://umap-learn.readthedocs.io/en/latest/), [scikit-learn](https://scikit-learn.org/stable/), [Janitor](https://github.com/pyjanitor-devs/pyjanitor/), [Plotly](https://plotly.com/python/)

The next workflow summarizes the steps performed by this method:
    
    
""")


image = Image.open('SOMoC_workflow.png')
st.image(image, caption='Clustering Workflow')

st.markdown("""
         **To cite the application, please reference XXXXXXXXX**
         """)



########### OPTIONS #######
# SIDEBAR

# Loading file
st.sidebar.header('Upload your SMILES')

uploaded_file = st.sidebar.file_uploader("Upload a TXT file with one SMILES per line", type=["txt"])


clustering_setting = st.sidebar.checkbox('Check to change the default configuration')
if clustering_setting == True:
    # Fingerprints
    st.sidebar.header('Fingerprint')    
    radius = st.sidebar.slider('Radius', 2, 6, 6, 1)
    nbits = st.sidebar.slider('Nº of bits', 512, 2048, 2048,512)
    kind = st.sidebar.selectbox("kind", ("counts", "bits"),0)
    # UMAP
    # https://umap-learn.readthedocs.io/en/latest/index.html
    st.sidebar.header('UMAP')    
    n_neighbors = st.sidebar.slider('Nº of neighbors', 5, 30, 25, 1)
    min_dist = st.sidebar.slider('Min distance', 0.0, 0.95, 0.0, 0.05)
    n_components = st.sidebar.slider('Nº of components', 2, 20, 10, 1)
    random_state = st.sidebar.slider('Random state', 0, 100, 10, 1)
    metric = st.sidebar.selectbox("metric", ("euclidean", "manhattan","canberra", "mahalanobis","cosine", "hamming","jaccard"),4)

    # GMM   
    min_n_clusters = st.sidebar.slider('Min nº of clusters', 2, 10, 2,1)
    max_n_clusters = st.sidebar.slider('Max nº of clusters', 10, 200, 50, 1)
    n_clusters =  np.arange(min_n_clusters,max_n_clusters)
    iterations = st.sidebar.slider('Iteration', 2, 20, 10, 1)
    n_init = st.sidebar.slider('Nº initiation', 2, 20, 10, 1) 
    init_params = st.sidebar.selectbox("init params", ("kmeans", "random"),0)
    covariance_type = st.sidebar.selectbox("covariance type", ("full", "tied","diag","spherical"),0)
    warm_start =  False
 

# Default configuration:    
else:   
    radius = 6
    nbits = 2048
    kind = "counts"
    n_neighbors = 25
    min_dist = 0.0
    n_components = 10
    random_state = 10
    metric = "cosine"
    min_n_clusters = 2
    max_n_clusters = 50    
    n_clusters= np.arange(min_n_clusters,max_n_clusters) # Range of K values to explore
    iterations= 10 # Iterations of GMM to run for each K
    n_init = 10 # Number of initializations on each GMM run, then just keep the best one.
    init_params = 'kmeans' # How to initialize. Can be random or K-means
    covariance_type = 'full' # Tipe of covariance to consider
    warm_start =  False


# Options for final clustering
ready = st.sidebar.checkbox('Check only if you have already decided the optimal k for clustering')

if ready == True:
    N_CLUST = st.sidebar.number_input('Nº of clusters (optimal k)', min_n_clusters, max_n_clusters, min_n_clusters,1)



#%%
# Fingerprints
def fingerprints_calculator(data):
    st.write('='*50)
    time_start = time.time()
    st.markdown("**Step 1: Calculating fingerprints**")
    # data = pd.read_csv(uploaded_file, sep='\t', delimiter=None, header=None, names=None)
    data_clase = data.copy()
    try:
        data_clase['mol'] = data_clase[0].apply(lambda x: Chem.MolFromSmiles(x))
        morgans = janitor.chemistry.morgan_fingerprint(data_clase, mols_column_name='mol',
                    radius=radius, nbits=nbits, kind=kind)
    except:
        st.error("**Oh no! There is a problem with fingerprint calculation of some smiles.**  :confused:")
        st.markdown(" :point_down: **Try using our standarization tool before clustering **")
        st.write("[LIDeB Standarization tool](https://share.streamlit.io/capigol/lbb-game/main/juego_lbb.py)")
        st.stop()
    nunique = morgans.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    morgans.drop(cols_to_drop, axis=1,inplace=True)
    X=morgans.values # X data, fingerprints values as a np array
    st.write(f'ALL DONE ! Fingerprints calculation took {round(time.time()-time_start)} seconds')
    st.write(f'{str(nbits)} bits have been generated..')
    st.write(f'{len(cols_to_drop)} constant bits have been dropped..')
    st.write('='*50)
    return X, data

#%%
# Dimensionality reduction by UMAP
def umap_reduction(X):
    st.markdown("**Step 2: Dimensionality reduction**")
    st.write('Reducing feature space with UMAP. Please wait...')
    time_start = time.time()
    UMAP_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric,
                                random_state=random_state).fit(X)
    embeddings = UMAP_reducer.transform(X)
    st.write(f'ALL DONE ! UMAP clustering took {round(time.time()-time_start)} seconds.')
    st.write(f'{str(embeddings.shape[1])} features have been selected.')
    st.write('='*50)    
    return embeddings


#%%
# GMM
def gmm(embeddings):
    st.markdown("**Step 3: Clustering**")
    st.write(f'Running GMM clustering for {len(n_clusters)} iterations..')
    time_start = time.time()
    temp = []
    lenght_dataset = len(embeddings)
    for n in n_clusters:
        if n < lenght_dataset:
            for x in range(iterations):
                gmm=GMM(n, n_init=n_init, init_params=init_params, covariance_type=covariance_type,
                            warm_start=warm_start, random_state=x, verbose=0).fit(embeddings) 
                labels=gmm.predict(embeddings)
                sil = silhouette_score(embeddings, labels, metric=metric)
                bic = gmm.bic(embeddings)
                df = pd.DataFrame({'Clusters':n, 'Silhouette': sil,'BIC': bic}, index=range(1))
                temp.append(df)
    results = pd.concat(temp, axis=0)
    st.write(f'ALL DONE ! GMM clustering took {round(time.time()-time_start)} seconds')
    st.write('='*50)   
    return results

def gmm1(embeddings):
    st.markdown("**Step 3: Clustering**")
    st.write(f'Running GMM clustering for {len(n_clusters)} iterations..')
    time_start = time.time()
    temp = []
    lenght_dataset = len(embeddings)
    for n in n_clusters:
        temp_sil = []
        temp_bic = []
        if n < lenght_dataset:
            for x in range(iterations):
                
                gmm=GMM(n, n_init=n_init, init_params=init_params, covariance_type=covariance_type,
                           warm_start=warm_start,random_state=x, verbose=0).fit(embeddings) 
                labels=gmm.predict(embeddings)
                sil = silhouette_score(embeddings, labels, metric=metric)
                bic = gmm.bic(embeddings)
                
                temp_sil.append(sil)
                temp_bic.append(bic)
    
            array_sil = np.array(temp_sil)
            sil_vuelta = array_sil.mean()
            sil_vuelta_sd = array_sil.std()
    
            array_bic = np.array(temp_bic)
            bic_vuelta = array_bic.mean()
            bic_vuelta_sd = array_bic.std()
    
            df_sil_bic = pd.DataFrame({'Clusters':n, 'Silhouette': round(sil_vuelta, 3), 'Sil_sd': round(sil_vuelta_sd, 3),'BIC': round(bic_vuelta, 3), 'BIC_sd': round(bic_vuelta_sd, 3)}, index=range(1))
            temp.append(df_sil_bic)
    
    results = pd.concat(temp, axis=0)
    
    st.write(f'ALL DONE ! GMM clustering took {round(time.time()-time_start)} seconds')
    st.write('='*50)   
    return results

# In[12]:
# ## Plotting

# By plotting the quality of the clustering as a function of the number of clusters, the optimal value of K can be identified as an inflection point in the curve, similar to the elbow method (Zhao et al., 2008; Bholowalia and Kumar, 2014).\
# SIL score is bounded [-1,1], while for BIC score the smaller the better.\
# BIC tends to be less conservative than SIL with respect to the number of clusters

def evaluation_plot(results):
    st.markdown("**Step 4: Clustering evaluation**")
    sns.set_context("talk", font_scale=1.1)#, rc={"lines.linewidth": 2.})
    sns.set_style("white")
    fig, ax1 = plt.subplots(figsize = (14, 6))
    ax2 = ax1.twinx()
    results.reset_index()

    sil=sns.lineplot(x='Clusters', y="Silhouette", data=results, color='b', ci='sd', err_style="bars", label = 'Silhouette',ax=ax1)
    plt.tick_params(labelsize=15)
    bic=sns.lineplot(x='Clusters', y="BIC", data=results, color='g', ci='sd', estimator=np.median,
                  err_style="bars", label='BIC',ax=ax2)
    sil.legend(fancybox=True,framealpha=0.5,fontsize='15',loc='center right', title_fontsize='30')
    bic.legend(fancybox=True,framealpha=0.5,fontsize='15',loc=(0.89,0.35), title_fontsize='30')
    plt.title("Selection of best k for Clustering", fontsize=20)
    plt.xlabel("Number of clusters (k)", fontsize=15)
    fig.tight_layout()
    bic.autoscale(enable=True, axis='both', tight=False); sil.autoscale(enable=True, axis='both', tight=False)
    sns.despine(ax=ax1,right=True, left=True); sns.despine(ax=ax2,right=True, left=True)
    st.markdown('**You can select the optimal value of K in the plot. The optimal value of K can be identified as an inflection point in the curve.**')
    st.write("Silhouette score is bounded [-1,1], the closer to one the better, while for BIC score the smaller the better.")
    st.write("BIC tends to be less conservative than silhouette with respect to the number of clusters")
    st.pyplot(fig)
    # st.write('='*50)

def evaluation_plot1(results):
    st.markdown("**Step 4: Clustering evaluation**")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=results['Clusters'], y=results['Silhouette'], 
                            mode='lines+markers', name= 'Silhouette', 
                            error_y = dict( type='data', symmetric=True, array= results["Sil_sd"]),
                            hovertemplate = "Clusters = %{x}<br>Silhouette = %{y}"), 
                            secondary_y=False)
    
    fig.add_trace(go.Scatter(x=results['Clusters'], y=results['BIC'], 
                            mode='lines+markers', name= 'BIC', 
                            error_y = dict( type='data', symmetric=True, array=results["BIC_sd"]),
                            hovertemplate = "Clusters = %{x}<br>BIC = %{y}"), 
                            secondary_y=True)
    
    fig.update_layout(title = "Silhouette coefficient and BIC vs K", title_x=0.5,
                  title_font = dict(size=28, family='Calibri', color='black'),
                  legend_title_text = "Metric", 
                  legend_title_font = dict(size=18, family='Calibri', color='black'),
                  legend_font = dict(size=15, family='Calibri', color='black'))
    fig.update_xaxes(title_text='K (Number of clusters)', range = [min_n_clusters - 0.5, max_n_clusters + 0.5],
                     tickfont=dict(family='Arial', size=16, color='black'),
                     title_font = dict(size=25, family='Calibri', color='black'))
    fig.update_yaxes(title_text='Silhouette coefficient', 
                     tickfont=dict(family='Arial', size=16, color='black'),
                     title_font = dict(size=25, family='Calibri', color='black'), secondary_y=False)
    fig.update_yaxes(title_text='BIC', 
                     tickfont=dict(family='Arial', size=16, color='black'),
                     title_font = dict(size=25, family='Calibri', color='black'), secondary_y=True)

    fig.update_layout(margin = dict(t=60,r=20,b=20,l=20), autosize = True)
    
    st.markdown('**You can select the optimal value of K in the plot. The optimal value of K can be identified as an inflection point in the curve.**')
    st.write("Silhouette score is bounded [-1,1], the closer to one the better, while for BIC score the smaller the better.")
    st.write("BIC tends to be less conservative than silhouette with respect to the number of clusters")
    st.plotly_chart(fig)


#%%

### Assign membership and save the dataset

def cluster_selection(uploaded_file,N_CLUST):
    st.markdown("**Step 4: Final clustering**")
    st.write(f'Clustering with K = {N_CLUST}')
    GMM_final = GMM(N_CLUST, n_init=n_init, init_params=init_params, warm_start=warm_start, covariance_type=covariance_type, random_state=random_state, verbose=0)
    GMM_final.fit(embeddings) 
    labels_final = GMM_final.predict(embeddings)
    st.write(f'GMM converged: {GMM_final.converged_}')

    sil_ok = round(float(silhouette_score(embeddings, labels_final,metric=metric)),4)
    # bic_ok = round(GMM_final.bic(embeddings),4)
    db_score = round(davies_bouldin_score(embeddings, labels_final),4)
    ch_score = round(calinski_harabasz_score(embeddings, labels_final),4)
    dist_dunn = pairwise_distances(embeddings)
    dunn_score = round(float(dunn(dist_dunn, labels_final)),4)
    
    validation_round = [sil_ok, db_score, ch_score, dunn_score]

    sil_random, sil_random_st, db_random, db_random_st, ch_random, ch_random_st, dunn_random, dunn_random_st = cluster_random(embeddings)
    
    random_means = [sil_random,db_random,ch_random,dunn_random]
    random_sds = [sil_random_st,db_random_st,ch_random_st,dunn_random_st]
    
    table_metrics = pd.DataFrame([validation_round,random_means,random_sds]).T
    table_metrics=table_metrics.rename(index={0: 'Silhouette score', 1:"Davies Bouldin score", 2: 'Calinski Harabasz score', 3:'Dunn Index'},columns={0:"Value",1:"Mean Random",2:"SD Random"})

    st.write(table_metrics)
    st.markdown(filedownload4(table_metrics), unsafe_allow_html=True)
    st.markdown("-------------------")

    validation_metrics = [sil_ok,db_score,ch_score,dunn_score]
    cluster_final = pd.DataFrame({'cluster': labels_final}, index=uploaded_file.index)
    data_clustered = uploaded_file.join(cluster_final)

    return data_clustered, validation_metrics


#%%
### Random cluster evaluations ###

def cluster_random(embeddings):
    compilado_silhoutte = []
    compilado_db = []
    compilado_ch = []
    compilado_dunn = []
    
    for i in range(100):
        random.seed(a=i, version=2)
        random_clusters = []
        for x in list(range(len(embeddings))):
            random_clusters.append(random.randint(0,N_CLUST-1))
        silhouette_random = silhouette_score(embeddings, np.ravel(random_clusters))
        compilado_silhoutte.append(silhouette_random)
        db_random = davies_bouldin_score(embeddings, np.ravel(random_clusters))
        compilado_db.append(db_random)
        ch_random = calinski_harabasz_score(embeddings, np.ravel(random_clusters))
        compilado_ch.append(ch_random)
        dist_dunn = pairwise_distances(embeddings)
        dunn_randome = dunn(dist_dunn, np.ravel(random_clusters))
        compilado_dunn.append(dunn_randome)

    sil_random = round(float(np.mean(compilado_silhoutte)),4)
    sil_random_st = round(np.std(compilado_silhoutte),4)
    db_random = round(mean(compilado_db),4)
    db_random_st = round(stdev(compilado_db),4)
    ch_random = round(mean(compilado_ch),4)
    ch_random_st = round(stdev(compilado_ch),4)
    dunn_random = round(float(mean(compilado_dunn)),4)
    dunn_random_st = round(np.std(compilado_dunn),4)

    return sil_random, sil_random_st, db_random, db_random_st, ch_random, ch_random_st, dunn_random, dunn_random_st


#%%
def distribution_plot(data_clustered):
    dataframe_final_1 = data_clustered["cluster"].value_counts().to_frame()
    dataframe_final_1['index_col'] = dataframe_final_1.index
    sns.set_theme(style="whitegrid")
    fig_2 = plt.figure()
    ax_2 = fig_2.add_axes([0,0,1,1])
    ax_2 = sns.barplot(x = dataframe_final_1['index_col'], y = dataframe_final_1["cluster"], data=dataframe_final_1,  palette="deep")
    plt.xlabel("Cluster")
    plt.ylabel("Nº of members by cluster")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    return ax_2

    
#%% Seetings
def setting_info():

    today = date.today()
    fecha = today.strftime("%d/%m/%Y")
    settings = []
    settings.append(["Date clustering was performed: " , fecha])
    settings.append(["Seetings:",""])
    settings.append(["",""])   
    settings.append(["Fingerprints",""])    
    settings.append(["radius:", str(radius)])
    settings.append(["nbits:", str(nbits)])
    settings.append(["kind:", str(kind)])
    settings.append(["",""])   
    settings.append(["UMAP",""])        
    settings.append(["n_neighbors:", str(n_neighbors)])
    settings.append(["min_dist:", str(min_dist)])
    settings.append(["n_components:", str(n_components)])
    settings.append(["random_state:", str(random_state)])
    settings.append(["metric:", str(metric)])
    settings.append(["",""])       
    settings.append(["GMM",""])        
    settings.append(["min_n_clusters:", str(min_n_clusters)])
    settings.append(["max_n_clusters:", str(max_n_clusters)])
    settings.append(["iterations:", str(iterations)])
    settings.append(["n_init:", str(n_init)])
    settings.append(["init_params",str(init_params)])
    settings.append(["covariance_type",str(covariance_type)])
    # settings.append(["",""])       
    # if ready == True:
    #     settings.append(["Results",""])
    #     settings.append(["selected K:",str(N_CLUST)])
    #     settings.append(["Silhouette coeficient:",str(validation_metrics[0])])
    #     # settings.append(["BIC:",str(validation_metrics[1])])
    #     settings.append(["Davies Bouldin score:",str(validation_metrics[1])])
    #     settings.append(["Calinski Harabasz score:",str(validation_metrics[2])])
    #     settings.append(["Dunn score:",str(validation_metrics[3])])
       
    settings.append(["",""])           
    settings.append(["To cite the application, please reference: ","XXXXXXXXXXX"])   
    settings_df = pd.DataFrame(settings)
    
    return settings_df

#%%
### Exporting files ###

def filedownload(df):
    csv = df.to_csv(index=True,header=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="cluster_assignations.csv">Download CSV File with the cluster assignations</a>'
    return href

def filedownload2(df):
    csv = df.to_csv(index=False,header=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="clustering_settings.csv">Download CSV File with your clustering settings</a>'
    return href

def filedownload4(df):
    csv = df.to_csv(index=True,header=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="Validations.csv">Download CSV File with the validation metrics</a>'
    return href


### Running ###

if uploaded_file is not None:
    run = st.button("RUN")
    if run == True:
        if ready == False:
            data = pd.read_csv(uploaded_file, sep='\t', delimiter=None, header=None, names=None)
            X = fingerprints_calculator(data)
            if n_components >= len(X[0]) -1:    
                st.error("Nº of components (umap) should be lower than the number of molecules less 1")
                st.stop()
            embeddings = umap_reduction(X[0])
            results = gmm1(embeddings)
            evaluation_plot1(results)
            st.markdown('**Once you have identified the optimal value of K, re-run the clustering but checking the option of "optimal k" **')
        if ready == True:
            data = pd.read_csv(uploaded_file, sep='\t', delimiter=None, header=None, names=None)
            X = fingerprints_calculator(data)
            embeddings = umap_reduction(X[0])
            data_clustered, validation_metrics = cluster_selection(X[1],N_CLUST)
            st.markdown(":point_down: **Here you can dowload the cluster assignations**", unsafe_allow_html=True)
            st.markdown(filedownload(data_clustered), unsafe_allow_html=True)
            st.markdown("-------------------")

            
            st.markdown(":point_down: **Here you can see the cluster distribution**", unsafe_allow_html=True)
            distribution_plot(data_clustered)
            
            settings_df = setting_info()
            st.markdown(":point_down: **Here you can download your settings**", unsafe_allow_html=True)
            st.markdown(filedownload2(settings_df), unsafe_allow_html=True)
            
else:
    st.info('Awaiting for TXT file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        if ready == False:
            data = pd.read_csv("molecules_1.txt", sep='\t', delimiter=None, header=None, names=None)
            X = fingerprints_calculator(data)
            if n_components >= len(X[0]) -1:    
                st.error("Nº of components (umap) should be lower than the number of molecules less 1")
                st.stop()
            embeddings = umap_reduction(X[0])
            results = gmm1(embeddings)
            evaluation_plot1(results)
            st.markdown('**Once you have identified the optimal value of K, re-run the clustering but checking the option of "optimal k" **')
        if ready == True:
            data = pd.read_csv("molecules_1.txt", sep='\t', delimiter=None, header=None, names=None)
            X = fingerprints_calculator(data)
            embeddings = umap_reduction(X[0])
            data_clustered, validation_metrics = cluster_selection(X[1],N_CLUST)
            st.markdown(":point_down: **Here you can dowload the cluster assignations**", unsafe_allow_html=True)
            st.markdown(filedownload(data_clustered), unsafe_allow_html=True)
            st.markdown("-------------------")

            
            st.markdown(":point_down: **Here you can see the cluster distribution**", unsafe_allow_html=True)
            distribution_plot(data_clustered)
            
            settings_df = setting_info()
            st.markdown(":point_down: **Here you can download your settings and validation metrics**", unsafe_allow_html=True)
            st.markdown(filedownload2(settings_df), unsafe_allow_html=True)


#Footer edit

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Made in  🐍 and <img style='display: ; ' href="https://streamlit.io" src="https://i.imgur.com/iIOA6kU.png" target="_blank"></img> Developed with ❤️ by <a style='display:; text-align: center;' href="https://lideb.biol.unlp.edu.ar/" target="_blank">LIDeB</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)






