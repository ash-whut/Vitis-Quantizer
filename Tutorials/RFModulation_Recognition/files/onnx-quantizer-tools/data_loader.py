import h5py as h5
import numpy as np
from sklearn.model_selection import train_test_split

def usable_snr_data(X, Y, Z):
    usable_snr_indices = np.where(np.any(Z >= 6, axis=1))
    return X[usable_snr_indices], Y[usable_snr_indices], Z[usable_snr_indices]

def best_snr_data(X, Y, Z):
    usable_snr_indices = np.where(np.any(Z == 30, axis=1))
    return X[usable_snr_indices], Y[usable_snr_indices], Z[usable_snr_indices]    

def data_loader(usable_snr=False, best_snr=False):
    data_file = '/home/ashwin/datasets/RADIOML_2021_07_INT8/RADIOML_2021_07_INT8.hdf5'
    # '/home/ashwin/datasets/RADIOML_2021_07_INT8/RADIOML_2021_07_INT8.hdf5'
    file_handle = h5.File(data_file,'r+')
    
    myData = file_handle['X'][:]  #1024x2 samples 
    myMods = file_handle['Y'][:]  #mods 
    mySNRs = file_handle['Z'][:]  #snrs 
    
    file_handle.close()
    np.random.seed(0)

    if usable_snr and best_snr:
        raise ValueError("Only one of 'usable_snr' or 'best_snr' can be True.")
    
    if usable_snr:
        myData, myMods, mySNRs = usable_snr_data(myData, myMods, mySNRs)
    elif best_snr:
        myData, myMods, mySNRs = best_snr_data(myData, myMods, mySNRs)

    myData = myData.reshape(myData.shape[0], 1024, 1, 2) 
    
    X_train ,X_test ,Y_train ,Y_test, Z_train, Z_test = train_test_split(myData, myMods, mySNRs, test_size=0.2, random_state=0)
    del myData, myMods, mySNRs
    X_val, X_test, Y_val, Y_test, Z_val, Z_test = train_test_split(X_test, Y_test, Z_test, test_size=0.5, random_state=0)

    returned_data = {
        "X_train": X_train,
        "X_test": X_test,
        "X_val": X_val, 
        "Y_train": Y_train,
        "Y_test": Y_test,
        "Y_val": Y_val, 
        "Z_train": Z_train,
        "Z_test": Z_test,
        "Z_val": Z_val
    }
    
    return returned_data  