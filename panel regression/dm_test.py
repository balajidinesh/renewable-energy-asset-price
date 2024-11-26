# Author   : John Tsang
# Date     : December 7th, 2017
# Purpose  : Implement the Diebold-Mariano Test (DM test) to compare 
#            forecast accuracy
# Input    : 1) actual_lst: the list of actual values
#            2) pred1_lst : the first list of predicted values
#            3) pred2_lst : the second list of predicted values
#            4) h         : the number of stpes ahead
#            5) crit      : a string specifying the criterion 
#                             i)  MSE : the mean squared error
#                            ii)  MAD : the mean absolute deviation
#                           iii) MAPE : the mean absolute percentage error
#                            iv) poly : use power function to weigh the errors
#            6) poly      : the power for crit power 
#                           (it is only meaningful when crit is "poly")
# Condition: 1) length of actual_lst, pred1_lst and pred2_lst is equal
#            2) h must be an integer and it must be greater than 0 and less than 
#               the length of actual_lst.
#            3) crit must take the 4 values specified in Input
#            4) Each value of actual_lst, pred1_lst and pred2_lst must
#               be numerical values. Missing values will not be accepted.
#            5) power must be a numerical value.
# Return   : a named-tuple of 2 elements
#            1) p_value : the p-value of the DM test
#            2) DM      : the test statistics of the DM test
##########################################################
# References:
#
# Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of 
#   prediction mean squared errors. International Journal of forecasting, 
#   13(2), 281-291.
#
# Diebold, F. X. and Mariano, R. S. (1995), Comparing predictive accuracy, 
#   Journal of business & economic statistics 13(3), 253-264.
#
##########################################################



import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import tensorflow as tf
from scipy.special import expit
import pickle

# from sklearn.model_selection import train_test_split
# from scipy.special import expit
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# from keras.layers import LSTM, Dense, Bidirectional, Flatten, TimeDistributed
# from tensorflow.keras.optimizers import Adam




def dm_test(actual_lst, pred1_lst, pred2_lst, h = 1, crit="MSE", power = 2):
    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt,msg)
        # Check the range of h
        if (h < 1):
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt,msg)
        len_act = len(actual_lst)
        len_p1  = len(pred1_lst)
        len_p2  = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return (rt,msg)
        # Check range of h
        if (h >= len_act):
            rt = -1
            msg = "The number of steps ahead is too large."
            return (rt,msg)
        # Check if criterion supported
        if (crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly"):
            rt = -1
            msg = "The criterion is not supported."
            return (rt,msg)  
        # Check if every value of the input lists are numerical values
        from re import compile as re_compile
        comp = re_compile("^\d+?\.\d+?$")  
        def compiled_regex(s):
            """ Returns True is string is a number. """
            if comp.match(s) is None:
                return s.isdigit()
            return True
#         for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
#             is_actual_ok = compiled_regex(f"{(abs(actual))}")
#             is_pred1_ok = compiled_regex(f"{(abs(pred1))}")
#             is_pred2_ok = compiled_regex(f"{(abs(pred2))}")
#             if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):  
#                 msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
#                 rt = -1
#                 return (rt,msg)
        return (rt,msg)
    
    # Error check
    error_code = error_check()
    # Raise error if cannot pass error check
    if (error_code[0] == -1):
        raise SyntaxError(error_code[1])
        return
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np
    
    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst  = []
    
    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()
    
    # Length of lists (as real numbers)
    T = float(len(actual_lst))
    
    # construct d according to crit
    if (crit == "MSE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((actual - p1)**2)
            e2_lst.append((actual - p2)**2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs((actual - p1)/actual))
            e2_lst.append(abs((actual - p2)/actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(((actual - p1))**(power))
            e2_lst.append(((actual - p2))**(power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)    
    
    # Mean of d        
    mean_d = pd.Series(d_lst).mean()
    
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    
    
    gamma = []
    for lag in range(0,h):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T + 1e-8
    DM_stat=V_d**(-0.5)*mean_d
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
    
    # Construct named tuple for return
#     dm_return = collections.namedtuple('dm_return', 'DM p_value')
    
#     rt = dm_return(DM = DM_stat, p_value = p_value)
    
    rt = {
        'DM' : DM_stat,
        'p_value' : p_value
    }
    
    return rt




def sequences(X,y,timesteps) :
    X = np.asarray(X)
    y = np.asarray(y)
    alpha = []
    beta = []
    n = timesteps
    for i in range(X.shape[0]): 
        if i < n-1 : 
            continue 
        alpha.append(X[i-(n-1):i+1])
        beta.append(y[i])
    
    return np.asarray(alpha),np.asarray(beta)

def root_mean_squared_loss(y_true,y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))



def get_cnn_lstm_model_vol(inputshape,kernel_size = 3): 
    (Timesteps,No_Features) = inputshape
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=kernel_size, activation='linear', input_shape=(Timesteps,No_Features)))
    model.add(MaxPooling1D(pool_size=1))
    model.add(LSTM(16, activation='linear', return_sequences=False))  # Use LSTM layer
    model.add(Flatten())
    model.add(Dense(16, activation='linear'))
    model.add(Dense(8, activation='linear'))
    model.add(Dense(1, activation='linear')) 

    model.compile(optimizer='adam', loss=root_mean_squared_loss, metrics=['mae'])
    model.summary()
    return model


def get_cnn_lstm_model_ret(inputshape,kernel_size=3): 
    (Timesteps,No_Features) = inputshape
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=kernel_size, activation='linear', input_shape=(Timesteps,No_Features)))
    model.add(MaxPooling1D(pool_size=1))
    model.add(LSTM(8, activation='linear', return_sequences=False))  # Use LSTM layer
    model.add(Flatten())
    model.add(Dense(8, activation='linear'))
    model.add(Dense(4, activation='linear'))
    model.add(Dense(1, activation='linear')) 

    model.compile(optimizer='adam', loss=root_mean_squared_loss, metrics=['mae'])
    model.summary()
    return model



def train_model(model, X , Y,window = 10, gap = 250,train_window=250 ,epochs = 10 ,valid_epochs =50) : 
    size = X.shape[0]
    final=[]
    
    
    for i in range(window) : 
        model.fit(X[:],Y[:], epochs=epochs , verbose=False)
        result = []
        print(i,end=': ')
        for w in range(size-gap,size,10):
            X_train ,y_train = X[(w-(train_window)):w],Y[(w-(train_window)):w]
            model.fit(X_train, y_train, epochs=valid_epochs , verbose=0)
            test = X[w:w+10] 
            preds = model.predict(test,verbose = 0)
            result.extend(preds)
            print(w,end=' ')
        print()
        final.append(result)
        rmse_1 = root_mean_squared_loss(Y[-250:].reshape(-1),((np.array(final)[-1]).reshape(-1)))
        rmse_full = root_mean_squared_loss(Y[-250:].reshape(-1),(np.mean(np.array(final),axis=0).reshape(-1)))
        x = [rmse_1,rmse_full]
        print(f"final : {x[0]} , meaned = {x[1]}")
        
    final = np.array(final)
    y_mean = np.mean(final,axis=0)
    
    return [y_mean,final],model



def plot_double_standard(Model,date,actual,sent,no_sent,sent_color='blue',no_sent_color='r',font_size=28,figsize=(30,10),fontfamily='serif',y_label='Log-Return'):
    time = np.asarray(pd.to_datetime(date))
    plt.figure(figsize=(30, 10)) 
    plt.rcParams['font.family'] = fontfamily
    plt.rcParams['font.size'] = font_size

    # Plotting the graph
    plt.plot(time, np.asarray(actual).reshape(-1,1),'grey', label='Actual',linewidth=2, dashes=(9, 3))  # Dotted red line for y_actual
    
    plt.plot(time, no_sent, no_sent_color, label=f'{Model}',linewidth=3) 
    plt.plot(time, sent , sent_color, label=f'{Model} Sentiment',linewidth=5) 

    # plt.ylim(-0.05,0.05)
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    # plt.subplots_adjust(bottom=-0.75)
    # plt.tight_layout(pad=2.0)

    plt.rcParams['font.size'] = 30
    plt.xlabel('Date',labelpad=20)
    plt.ylabel(y_label, labelpad = 20 )
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3,frameon=False)

#   saveplt.savefig(save_at, bbox_inches='tight')
    plt.show()  








