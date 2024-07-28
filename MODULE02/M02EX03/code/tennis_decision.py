import numpy as np

TENNIS = np.array([
    ["Sunny", "Hot", "High", "Weak", "No"],
    ["Sunny", "Hot", "High", "Strong", "No"],
    ["Overcast", "Hot", "High", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High", "Weak", "No"],
    ["Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "Normal", "Weak", "Yes"]
])

def compute_prior_probablity(train_data):
    y_unique = ['No', 'Yes']
    prior_probablity = np.zeros(len(y_unique))
    for i in range(len(y_unique)):
        prior_probablity[i] = len(np.where(train_data[:, 4] == y_unique[i])[0]) / len(train_data)
    return prior_probablity

def get_index_from_value(feature_name, list_name):
    return np.where(list_name == feature_name)[0][0]

def compute_conditional_probability(train_data):
    y_unique = ['No', 'Yes']
    conditional_probability = []
    list_x_name = []
    for i in range(0,train_data.shape[1]-1):
        x_unique = np.unique(train_data[:,i])
        list_x_name.append(x_unique)

        x_conditional_probability = np.zeros((len(y_unique), len(x_unique)))
        for j in range(len(y_unique)):
            for z in range(len(x_unique)):
                x_conditional_probability[j, z] = len(np.where((train_data[:, i] == x_unique[z]) & (train_data[:, 4] == y_unique[j]))[0]) / len(np.where(train_data[:, 4] == y_unique[j])[0])
        
        conditional_probability.append(x_conditional_probability)
    return conditional_probability, list_x_name

def train_naive_bayes(train_data):
    y_unique = ['No', 'Yes']
    prior_probability = compute_prior_probablity(train_data)
    conditional_probability, list_x_name = compute_conditional_probability(train_data)

    return prior_probability, conditional_probability, list_x_name


def predict(X, list_x_name, prior_probability, conditional_probability):
    x1 = get_index_from_value(X[0], list_x_name[0])
    x2 = get_index_from_value(X[1], list_x_name[1])
    x3 = get_index_from_value(X[2], list_x_name[2])
    x4 = get_index_from_value(X[3], list_x_name[3])

    p1 = prior_probability[0] \
    *conditional_probability[0][0, x1] \
    *conditional_probability[1][0, x2] \
    *conditional_probability[2][0, x3] \
    *conditional_probability[3][0, x4] \

    p2 = prior_probability[1] \
    *conditional_probability[0][1, x1] \
    *conditional_probability[1][1, x2] \
    *conditional_probability[2][1, x3] \
    *conditional_probability[3][1, x4] \
    
    if p1 > p2:
        y_pred = 0
    else:
        y_pred = 1

    return y_pred


def main():
    X = ['Sunny','Cool', 'High', 'Strong']
    prior_probability,conditional_probability, list_x_name = train_naive_bayes(TENNIS)
    pred = predict(X, list_x_name, prior_probability, conditional_probability)
    if(pred):
        print("Ad should go!")
    else:
        print("Ad should not go!")
if __name__ == '__main__':
    main()