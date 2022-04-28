import re

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = nltk.wordnet.WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
# Possible stop_word from tagline
stop_words.add("nan")

DATASET_PATH = "./dataset/"

def get_dataset(
    cols: list = ["id", "title", "overview", "tagline", "genres", "original_language", "poster_path"],
    language: str = "en",
    feature_cols: list = ["overview"],
    parse_genres: bool = False,
):
    all_movies = pd.read_csv(DATASET_PATH + "movies_metadata.csv", low_memory=False)

    # Filter columns
    all_movies = all_movies[cols]

    # Filter by language
    movies_by_language = all_movies[all_movies["original_language"] == language]

    # Get rid of original_language column
    out_cols = list(filter(lambda col: col!= "original_language", cols))
    movies_by_language = movies_by_language[out_cols]

    if parse_genres:
        parse_genres_col_name = "processed_genres"
        
        genres = list(map(lambda g: re.findall("'name':\s*'(\w*)'", g), movies_by_language["genres"]))
        joined_genres = list(map(lambda g: " ".join(g), genres))
        
        movies_by_language[parse_genres_col_name] = joined_genres

        feature_cols.append(parse_genres_col_name)

    movies_by_language["corpus"] = get_corpus_column(movies_by_language[feature_cols].astype(str).agg(' '.join, axis=1)) #.apply(get_pre_processed_text)

    print('Summary of dataset\nSize: {}\nFirst 10 rows of corpus:\n'.format(movies_by_language.shape[0]))
    print(movies_by_language["corpus"].head(10))

    return movies_by_language

def remove_non_ascii(text: str):
    """ Removes non-ASCII characters """
    chars = [char for char in text if ord(char) < 128]
    
    return "".join(chars)


def tokenize_and_lowercase(text: str):
    """ Tokenizes into lowercased tokens with Regex Tokenizer"""
    tokens = []
    sentences = sent_tokenize(text)

    for sentence in sentences:
        regex_tokenizer = re.compile("\w+[-'.]\w+|\w+")  # tokenizer removes whitespaces and punctuation
        tokenized = re.findall(regex_tokenizer, sentence)

        for sentence in tokenized:
            tokens.append(lemmatizer.lemmatize(sentence.lower()))
            # tokens.append(e.lower())

    return tokens


def remove_stop_words(
    tokenized_text: str, 
    stop_words = stop_words,
):
    """ Removes stop words """
    without_stop_words = [x for x in tokenized_text if x not in stop_words]

    return without_stop_words

def get_pre_processed_text(
    text: str = '', 
    stop_words = stop_words,
):
    only_ascii_text = remove_non_ascii(text)
    tokenized_lowercased_text = tokenize_and_lowercase(only_ascii_text)
    pre_processed_text = remove_stop_words(tokenized_lowercased_text, stop_words)
    
    return pre_processed_text

def get_corpus_column(
    df_col
):
    corpus = df_col.apply(remove_non_ascii)
    corpus = corpus.apply(tokenize_and_lowercase)
    corpus = corpus.apply(remove_stop_words)

    return corpus


def word_embeddings_vectorize(model, texts):
    embeddings = []
    
    for text in texts:
        vec = None
        count = 0

        for word in text:
            if word in model.wv.key_to_index:
                count += 1
                
                if vec is None:
                    vec = model.wv[word]
                else:
                    vec = vec + model.wv[word]
                
        if vec is not None:
            vec = vec / count
        
        embeddings.append(vec)
    
    return embeddings

def word_embeddings_predict(embedded, input_idx, n_outputs = 5):
    similarities = []

    for idx in range(len(embedded)):
        try:
            similarity = cosine_similarity(
                np.array(embedded[input_idx]).reshape(1, -1),
                np.array(embedded[idx]).reshape(1, -1)
            )

            similarities.append(similarity[0][0])
        except ValueError:
            similarities.append(0)

    out_idx_list = sorted(range(len(similarities)), key=lambda k: similarities[k], reverse=True)[:n_outputs+1][1:]
    outputs = [(idx, similarities[idx]) for idx in out_idx_list]
    
    return outputs