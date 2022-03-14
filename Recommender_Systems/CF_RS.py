### COLLABORATIVE-BASED FILTERING ###
"""
Pros:
+ No Hyperparameters
+

Cons:
- Popularity Bias
- Cold-start problem
- Grey/black sheep problem
"""

def CF_recommender(df, target, method='Cosine'):
    """ COLLABORATIVE FILTERING-BASED RECOMMENDATIONS
    (1) Create sparse Item-Item Matrix (=more stable, but non-personalized) or User-User Matrix(=less stable, but personalized)
    (2) Define similarity distance measure, i.e. Cosine, Conditional Probability, Bipartite Graph
    (3) RETURN sorted ranked list based on similarity score
    METHODS: Cosine, Conditional Probability, Bipartite Graph
    HYPERPARAMETERS: (1) popularity bias, (2) cold-start problem, (3) grey/black sheep
    """
    return
