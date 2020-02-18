import pickle
with open('/home/ubuntu/files/My_First_Git_Backed_Project/nyc_health_violation_rf_model.pkl', 'rb') as f:
    clf = pickle.load(f)

def endpoint(*inputs):
    return clf.predict(inputs[0])[0]
