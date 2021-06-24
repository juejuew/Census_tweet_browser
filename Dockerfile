FROM johanngb/rep-int:latest

### Prepare for installation
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN apt install -y gdebi-core 
WORKDIR /home/rep/

RUN /usr/bin/python3 -m pip install --upgrade pip

### Packages
#RUN R -e 'install.packages(c("tidyverse", "devtools", "reticulate", "renv"))'
RUN pip install dash dash_bootstrap_components dash_html_components dash_table dash_extensions sklearn spicy nltk umap umap-learn hdbscan chart_studio 

WORKDIR /home/rep/
COPY allCensus_sample.csv allCensus_sample.csv
COPY trust_condensed_sample_smaller.csv trust_condensed_sample_smaller.csv 
COPY state_populations.csv state_populations.csv
COPY dimRed_cluster_plot.html dimRed_cluster_plot.html
COPY app_with_advanced_options.py app_with_advanced_options.py
COPY distance_quote.ipynb distance_quote.ipynb
COPY tweet_brower_not_dash.ipynb tweet_brower_not_dash.ipynb

### Clean up after installation
RUN apt clean
RUN rm -rf /root/.cache
RUN rm -rf /tmp/*
ENV DEBIAN_FRONTEND teletype

WORKDIR /startup
CMD ["bash", "/startup/startup.sh"]