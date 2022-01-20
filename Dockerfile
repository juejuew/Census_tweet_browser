FROM johanngb/rep-int:latest

### Prepare for installation
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN apt install -y gdebi-core 
WORKDIR /home/rep/

RUN /usr/bin/python3 -m pip install --upgrade pip

### Packages
RUN pip install dash dash_bootstrap_components dash_html_components dash_table dash_extensions dash_uploader rpy2 sklearn spicy nltk umap umap-learn hdbscan chart leidenalg igraph _studio xlrd
RUN R -e "install.packages('dplyr',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('useful',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('tidyverse',dependencies=TRUE, repos='http://cran.rstudio.com/')" 
RUN R -e "install.packages('lubridate',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('ggplot2',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('stringr',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('reshape2',dependencies=TRUE, repos='http://cran.rstudio.com/')"

WORKDIR /home/rep/
COPY allCensus_sample.csv allCensus_sample.csv
COPY trust_condensed_sample_smaller.csv trust_condensed_sample_smaller.csv 
COPY state_populations.csv state_populations.csv
COPY dimRed_cluster_plot.html dimRed_cluster_plot.html
COPY app_with_advanced_options.py app_with_advanced_options.py
COPY distance_quote.ipynb distance_quote.ipynb
COPY tweet_brower_not_dash.ipynb tweet_brower_not_dash.ipynb


