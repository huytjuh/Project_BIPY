### CONTENT-BASED RECOMMENDER SYSTEM ###

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def TFIDF_recommender(df, target, method='cosine', language='english', **kwargs):
    """ CONTENT-BASED TF-IDF RECOMMENDATIONS
    (1) Calculate TF-IDF = [# terms i / # total terms] * Log([# total documents / # documents containing term i])
    (2) Similarity matrix, e.g. Cosine = Dot(d1, d2) / ||d1|| * ||d2||
    (3) RETURN sorted ranked list based on cosine similarity score
    METHODS: Cosine
    HYPERPARAMETERS: ngram_range (=before/after of word to see context), max_df (=too frequently), min_df (=too infrequently)
    """
    TFIDF = TfidfVectorizer(stop_words=language, **kwargs)
    TFIDF_matrix = TFIDF.fit_transform(df[[target]])

    if method=='cosine':
        similarity = linear_kernel(TFIDF_matrix, TFIDF_matrix)

    INDICES = pd.Series(df, index = movies['title']).drop_duplicates()

    return TFIDF_matrix
