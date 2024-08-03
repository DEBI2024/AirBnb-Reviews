
'''
This script contains the function to extract topics from the data using Latent Dirichlet Allocation (LDA) and BERTopic.

Author: Shahd Seddik
Date: 22-07-2024
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

def sample_polarity(data, polarity, sample_size):
    '''
    Sample a subset of the data with a specific polarity.
    '''
    polarity_data = data[data['polarity_class'] == polarity]
    sample_size = min(sample_size, polarity_data.shape[0])
    return polarity_data.sample(sample_size, random_state=42)

def get_topics_lda(data, n_topics, city):
    '''
    Apply LDA to extract topics from the data
    '''
    def print_topics_lda(lda, vectorizer, n_words=10):
        print("Topics for {}".format(city))
        for i, topic in enumerate(lda.components_):
            print(f'\tTopic {i}:')
            print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-n_words:]])
            print()

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(data['comments_preprocessed'])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf)

    # Print the top words for each topic
    print_topics_lda(lda, vectorizer)

    return lda, vectorizer

def get_topics_bert(df, city, nr_topics):
    '''
    Apply BERT to extract topics from the data
    '''
    print(f'Getting topics for {city}...')
    # Initialize BERTopic
    representation_model = KeyBERTInspired()
    topic_model = BERTopic('english',
                           representation_model=representation_model,
                           calculate_probabilities=False,
                           verbose=True,
                           nr_topics=nr_topics,
                           n_gram_range=(1, 2),
                           top_n_words=10)

    # Fit BERTopic
    topics = topic_model.fit_transform(df['comments_preprocessed'])

    return topic_model

