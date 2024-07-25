"""This module is used to download, extract, and split the data into training and testing datasets
The data is downloaded from Inside Airbnb website for the cities: London and NYC
The data is split into 80% training and 20% testing datasets then uploaded to Kaggle

Author: AbdElRhman ElMoghazy
Date 20-07-2024
"""
import os
import gzip
import shutil
import glob
import wget
import pandas as pd
import scipy.sparse.linalg
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi
from textblob import TextBlob


CITIES = ["london", "nyc"]
COLUMNS = ['id', 'listing_url', 'scrape_id', 'last_scraped', 'name',
       'neighborhood_overview', 'host_id', 'number_of_reviews',
       'number_of_reviews_ltm', 'number_of_reviews_l30d', 'first_review',
       'last_review', 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value', 'calculated_host_listings_count','reviews_per_month']

cities_reviews = {
    "nyc": "https://data.insideairbnb.com/united-states/" \
        "ny/new-york-city/2024-07-05/data/reviews.csv.gz",
    "london": "https://data.insideairbnb.com/united-kingdom/" \
        "england/london/2024-06-14/data/reviews.csv.gz"
}

cities_listings = {
    "nyc": "https://data.insideairbnb.com/united-states/ny/new-york-city/" \
        "2024-07-05/data/listings.csv.gz",
    "london": "https://data.insideairbnb.com/united-kingdom/" \
        "england/london/2024-06-14/data/listings.csv.gz"
}


def rename_file(city: str, kind: str) -> str:
    """renaming downloaded files to include kind and city names
    Args:
      city: name of the city to be downloaded
      kind: can be either listings or reviews

    Returns:
      new_name: new name of the file
    """

    current_name = f"./{kind}.csv.gz"
    new_name = f"./{kind}_{city}.csv.gz"
    os.rename(current_name, new_name)

    return new_name


def extract_cities(file_name: str):
    """extracting the .gz file
    Args:
      file_name: name of the file to be extracted
    """

    with gzip.open(file_name, 'rb') as f_in:
        with open(file_name[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # remove the original .gz file after extraction
    os.remove(file_name)


def get_city_data(city: str):
    """download and extract city data for both reviews and listings
    Args:
      city: name of the city to be downloaded and extracted
    """

    wget.download(cities_reviews[city])
    wget.download(cities_listings[city])

    for kind in ["reviews", "listings"]:
        file_name = rename_file(city, kind)
        extract_cities(file_name)


def get_cities_data():
    """download and extract all cities data"""

    for city in CITIES:
        get_city_data(city)


def get_file_names():
    """get the names of all csv files in the folder"""

    extension = 'csv'
    csv_files = glob.glob('*.{}'.format(extension))
    return csv_files


def save_data(data: str, path: str):
    """save the data to a specific path"""
    data.to_csv(path)


def calculate_scores(data: pd.DataFrame)-> pd.DataFame:
  """calculate the scores of the comments using textblob
  Args:
    data: contains the comments needed to calculate the score
  Returns:
    data: same object after adding the needed columns"""
  
  subjectivity = []
  polarity = []
  for comment in data.comments.astype(str):
      blob = TextBlob(comment)
      subjectivity.append(blob.sentiment.subjectivity)
      polarity.append(blob.sentiment.polarity)
  
  data["subjectivity"] = subjectivity
  data["polarity"] = polarity
  
  return data

def split_data(csv_files_names: list):
  """split the data and save it to the dataset folder to be ready for kaggle upload
  Args:
    csv_files_names: the list of all csv files in the folder
  """

  for city in CITIES:

    # paths to listings and reviews within for the city
    city_listings_path = [x for x in csv_files_names if x.startswith("listings") and city in x][0]
    city_reviews_path = [x for x in csv_files_names if x.startswith("reviews") and city in x][0]

    city_listings = pd.read_csv(city_listings_path)[COLUMNS]
    city_reviews = pd.read_csv(city_reviews_path)

    # add city name to help when merginig different cities in the future
    city_listings["city"] = city
    data_city = city_listings.rename({"id": "listing_id"}, axis = 1).merge(city_reviews, on = "listing_id", how = "inner")
    
    data_city = calculate_scores(data_city)

    # divide the data into 80% training and 20% testing datasets
    train, test = train_test_split(data_city, test_size=0.2, shuffle = True)

    save_data(train, f"./dataset/{city}_train.csv")
    save_data(test, f"./dataset/{city}_test.csv")

    # free memory
    del data_city
    del train
    del test


def main(is_dataset_new):
    """executing etl steps"""
    api = KaggleApi()
    api.authenticate()

    get_cities_data()
    csv_files_names = get_file_names()
    split_data(csv_files_names)

    for file in csv_files_names:
        os.remove(file)

    if is_dataset_new:
        api.dataset_create_new(folder="./dataset")
    else:
        api.dataset_create_version(folder="./dataset", version_notes="update")