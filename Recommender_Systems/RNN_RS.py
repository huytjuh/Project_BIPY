### DEEP LEARNING-BASED RECOMMENER SYSTEM ###

def GRU_recommender():
    """ SESSION-BASED RNN GATED RECURRENT UNIT (GRU) RECOMMENDATIONS
    (1) Designed to make predictions in data that comes in the form of a sequence
    (2) Takes one hot encoded item vector {0,1} as input, or weighted sum item vector for earlier events (discounted)
    (3) GRU layers: (=Get rid of cell state and use only hidden state instead)
        (a) Update gate F_t + I_t (= decides what information to throw away and what to add)
        (b) Reset gate (= how much past information to forget)
    (4) RETURN ietem of the next even in the session
    METHODS: loss function RMSE/MSE, L1/L2 regularization
    HYPERPARAMETERS: n_factors (=# latent factors), n_epochs (=#iterations SGD), lr_all (=learning rate), reg_all (=regularization terms)
    """   
    return

def LSTM_recommender():
    """ SESSION-BASED RNN LONG SHORT-TERM MEMORY RECOMMENDATIONS
    (1) Designed to make predictions in data that comes in the form of a sequence
    (2) Takes one hot encoded item vector {0,1} as input, or weighted sum item vector for earlier events (discounted)
    (3) LSTM layers: 
        (a) Cell state C_t (=memory); C_t = C_t-1 x F_t x C x I_t
        (b) Forget gate/layer F_t (=remove non-relevant data)
        (c) Candidate layer C (=holds possible value to add to the cell state)
        (d) Input gate/layer I_t (=evaluates what data from the candidate layer should be add to cell state)
        (e) Output gate/layer O_t (=computes the output)
        (f) Hidden state H_t = O_t x tanh(C_t)
    (4) RETURN item of the next even in the session
    METHODS: loss function RMSE/MSE, L1/L2 regularization
    HYPERPARAMETERS: n_factors (=# latent factors), n_epochs (=#iterations SGD), lr_all (=learning rate), reg_all (=regularization terms)
    """       
    return
