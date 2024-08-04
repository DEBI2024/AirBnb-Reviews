# Airbnb Reviews

For this project we have different components as follows:

# ETL:
used to extract the data form the website and load it locally and into Kaggle Private Dataset (feel free to contact the owners to add you to the dataset)

To run the etl use the following snippet `etl.main(is_dataset_new, upload)`
use `is_dataset_new` if you want to create a dataset and `upload` if you want to uplad a version

in the dataset folder a file called dataset-metadata.json should be added as follows:

```
{
  "title": "Airbnb cities reviews", 
  "id": "moghazy/airbnb-cities-reviews", 
  "licenses": [{"name": "CC0-1.0"}]
}

```

More examples can be found in the notebook `model_trial.ipynb`

# Models & Analysis:
1. GRU: `gru_model.py` results and analysis in `model_trial.ipynb`
2. REMBert: `airbnb_sota_sentiment_RemBert.ipynb`
3. DistilBert: `DistBERTandSpatial.ipynb`

## DistilBert Model Training

### Train the Model
The model is trained using the DistilBERT architecture with the following configuration:
- **Epochs:** 1
- **Batch Size:** 128
- **Learning Rate:** Optimized with weight decay

### Save the Model
After training, the model and tokenizer are saved to the directory `./fine-tuned-distilbert`.

# Topic Modeling:
Models and README.md for this specific part can be found in `topics` directory

# Libraries Used:
- pandas==2.2.2
- datasets==2.20.0
- transformers==4.42.3
- scikit-learn==1.2.2
- matplotlib==3.7.5
- seaborn==0.12.2
- geopandas==0.14.4
- folium==0.17.0
- nltk==3.8.1
- gensim==4.3.3
- vaderSentiment==3.3.2
- wget==3.2
- wordcloud==1.9.3
- kaggle==1.6.17
- kagglehub==0.2.8
- keras==3.4.1
- textblob==0.17.1
- tqdm==4.66.4
- pip install pandas==2.1.4
- scikit-learn==1.2.2 
- torch==2.3.1+cu12 
- transformers==4.42.4 
- datasets==2.20.0 
- beautifulsoup4 
- nltk==3.8.1 
- langdetect==1.0.9 
- matplotlib==3.8.0 
- seaborn==0.12.2 
- tqdm==4.66.4
- pip install pandas==2.1.4 
- numpy==1.26.4 
- matplotlib==3.8.0 
- seaborn==0.12.2 
- scikit-learn==1.2.2 
- wordcloud==1.9.3`
