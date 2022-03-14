### FACTOR-BASED RECOMMENDER SYSTEM ###

def ALS_recommender(df, target):
    """ FACTOR-BASED ALTERNATING LEAST SQUARES RECOMMENDATIONS
    (1) Create sparse User-Item Matrix in PySpark
    (2) Decompose Rating Matrix into User Matrix and Item Matrix, R = U x I
    (3) RETURN sorted ranked list based on Rating
    METHODS: loss function RMSE/MSE, L1/L2 regularization
    HYPERPARAMETERS: rank (=# of latent factors),  maxIter (=# alternating), regParam (=overfitting terms)
    """
    return

def SVD_recommender(df, target):
    """ FACTOR-BASED SINGULAR VALUE DECOMPOSITION RECOMMENDATIONS
    (1) Create non-sparse User-Item Matrix in PySpark
    (2) Decompose Rating Matrix using PCA to get (1) latent User-User features and (2) latent Item-Item features R=ASV
    (3) RETURN sorted ranked list based on Rating
    METHODS: loss function RMSE/MSE, L1/L2 regularization
    HYPERPARAMETERS: n_factors (=# latent factors), n_epochs (=#iterations SGD), lr_all (=learning rate), reg_all (=regularization terms)
    """
    return

def SVDpp_recommender(df, target):
    """ FACTOR-BASED SINGULAR VALUE DECOMPOSITION RECOMMENDATIONS
    (1) Create sparse User-Item Matrix, incl. implicit feedback, in PySpark
    (2) Decompose Rating Matrix using PCA to get (1) latent User-User features and (2) latent Item-Item features R=ASV
    (3) RETURN sorted ranked list based on Rating
    METHODS: loss function RMSE/MSE, L1/L2 regularization
    HYPERPARAMETERS: n_factors (=# latent factors), n_epochs (=#iterations SGD), lr_all (=learning rate), reg_all (=regularization terms)
    """
    return

def NMF_recommender(df, target):
    """ FACTOR-BASED SINGULAR VALUE DECOMPOSITION RECOMMENDATIONS
    (1) Create sparse User-Item Matrix in PySpark
    (2) Decompose Rating Matrix using PCA to get (1) latent User-User features and (2) latent Item-Item features; R=ASV A,V kept positive
    (3) RETURN sorted ranked list based on Rating
    METHODS: loss function RMSE/MSE, L1/L2 regularization
    HYPERPARAMETERS: n_factors (=# latent factors), n_epochs (=#iterations SGD), lr_all (=learning rate), reg_all (=regularization terms)
    """
    return

def lightFM_recommender(df, target):
    """ FACTOR-BASED HYBRID FACTORIZATION MACHINE RECOMMENDATIONS
    (1) Create spare one-hot-encoded tuples, incl. one-hot-encoded User and Item features, Auxiliary Features, Obs. Ratings
    (2) Define loss function and optimization algorithm
    (3) Convert Ratings back to User-Item Matrix for full information retrieval
    (4) RETURN sorted ranked list based on Rating
    METHODS: Logistic, Bayesian Personalised Ranking (BPR), Weighted Approximate-Rank Pairwise (WARP)
    HYPERPARAMETERS: no_components (=dimensionality), learning_schedule (=adagrad, adadelta), learning_rate, item_alpha (=L2 penalty), user_alpha (=L2 penalty), max_sampled, num_epochs 
    """
    return
