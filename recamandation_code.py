import joblib
 

import pickle
# Save the model as a pickle file
#with open('xgb_model.pkl', 'wb') as file:
#    pickle.dump(xgb, file)

# Load the model from the pickle file
with open('xgb_model.pkl', 'rb') as file:
    loaded_xgb = pickle.load(file)


knn_from_joblib = loaded_xgb
 

def recondation_fn(features_list):
    # Convert the list of strings to a list of floats
    features_list = [float(feature) for feature in features_list]
    
    print(features_list)
    
    import numpy as np
    
    # Convert list to numpy array
    int_features2 = np.array(features_list)
    
    # Reshape the array for the model input
    int_features1 = int_features2.reshape(1, -1)
    
    # Make the prediction using the pre-loaded model
    tested1 = knn_from_joblib.predict(int_features1)
    
    print(tested1)
    
    return tested1



