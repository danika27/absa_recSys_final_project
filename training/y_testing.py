import y_get_matrices
import numpy as np
import pandas as pd


# def get_MSE(U1, U2, H1, H2, V, A, X, Y, lambda_x, lambda_y, lambda_u, lambda_h, lambda_v):
#     e1 = ((U1.dot(U2.T) + H1.dot(H2.T) - A) ** 2).mean()
#     e2 = lambda_x * (((U1.dot(V.T) - X) ** 2).mean())
#     e3 = lambda_y * (((U2.dot(V.T) - Y) ** 2).mean())
#     e4 = lambda_u * ((U1 ** 2).mean() + (U2 ** 2).mean())
#     e5 = lambda_h * ((H1 ** 2).mean() + (H2 ** 2).mean())
#     e6 = lambda_v * ((V ** 2).mean())
#     return e1 + e2 + e3 + e4 + e5 + e6

def A_RMSE(A_dense, U1, U2, H1, H2):
    E = 0
    for k in range(len(A_dense)):
        i = int(A_dense[k, 0])
        j = int(A_dense[k, 1])
        r_ij = A_dense[k, 2]
        e = r_ij - U1[i, :].dot(U2.T[:, j]) - H1[i, :].dot(H2.T[:, j])
        E += e**2

    return (E/len(A_dense))**0.5


def X_RMSE(X_valid, X, U1, V):
    E = 0
    n = 0
    for i in range(len(X)):
        for j in range(len(X[i])):
            if X[i, j] == -1:
                e = X_valid[i, j] - U1[i, :].dot(V.T[:, j])
                n += 1
                E += e**2
    return (E/n)**0.5


def Y_RMSE(Y_valid, Y, U1, V):
    E = 0
    n = 0
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            if Y[i, j] == -1 and Y_valid[i, j] != 0:
                e = Y_valid[i, j] - U1[i, :].dot(V.T[:, j])
                n += 1
                E += e**2
    return (E/n)**0.5

def start_training(A_dense, A, X, Y, valid_set, r, r_, lambda_x, lambda_y, lambda_u, lambda_h, lambda_v, T):
    m = X.shape[0]
    p = X.shape[1]
    n = Y.shape[0]
    U1 = np.random.rand(m, r)
    U2 = np.random.rand(n, r)
    V = np.random.rand(p, r)
    H1 = np.random.rand(m, r_)
    H2 = np.random.rand(n, r_)
    t = 0
    while t <= T:
        t += 1
        tmp1 = lambda_x * (X.T.dot(U1)) + lambda_y * (Y.T.dot(U2))
        tmp2 = V.dot(lambda_x * U1.T.dot(U1) + lambda_y * U2.T.dot(U2) + lambda_v * np.eye(r))
        tmp3 = np.sqrt(np.divide(tmp1, tmp2))
        V = np.multiply(V, tmp3)
        # update V
        tmp1 = A.dot(U2) + lambda_x * X.dot(V)
        tmp2 = (U1.dot(U2.T) + H1.dot(H2.T)).dot(U2) + U1.dot(lambda_x * V.T.dot(V) + lambda_u * np.eye(r))
        tmp3 = np.sqrt(np.divide(tmp1, tmp2))
        U1 = np.multiply(U1, tmp3)
        # update U1
        tmp1 = A.T.dot(U1) + lambda_y * Y.dot(V)
        tmp2 = (U2.dot(U1.T) + H2.dot(H1.T)).dot(U1) + U2.dot(lambda_y * V.T.dot(V) + lambda_u * np.eye(r))
        tmp3 = np.sqrt(np.divide(tmp1, tmp2))
        U2 = np.multiply(U2, tmp3)
        # update U2
        tmp1 = A.dot(H2)
        tmp2 = (U1.dot(U2.T) + H1.dot(H2.T)).dot(H2) + lambda_h*H1
        tmp3 = np.sqrt(np.divide(tmp1, tmp2))
        H1 = np.multiply(H1, tmp3)
        # update H1
        tmp1 = A.T.dot(H1)
        tmp2 = (U2.dot(U1.T) + H2.dot(H1.T)).dot(H1) + lambda_h * H2
        tmp3 = np.sqrt(np.divide(tmp1, tmp2))
        H2 = np.multiply(H2, tmp3)
        # update H2
        error = A_RMSE(valid_set, U1, U2, H1, H2)
        print (error)
    return [U1, U2, V, H1, H2]

def training(A_dense, A, X, Y, valid_set,  r, r_, lambda_x, lambda_y, lambda_u, lambda_h, lambda_v, T, alpha, beta):
    m = X.shape[0]
    p = X.shape[1]
    n = Y.shape[0]
    U1 = np.random.uniform(-0.05, 0.05, (m, r))
    U2 = np.random.uniform(-0.05, 0.05, (n, r))
    V = np.random.uniform(-0.05, 0.05, (p, r))
    H1 = np.random.uniform(-0.05, 0.05, (m, r_))
    H2 = np.random.uniform(-0.05, 0.05, (n, r_))
    [X, X_valid] = y_get_matrices.get_X_validation(X)
    [Y, Y_valid] = y_get_matrices.get_X_validation(Y)
    t = 0
    prev_rmse = 100
    while t < T:
        t += 1
        _U1 = U1
        _U2 = U2
        _V = V
        _H1 = H1
        _H2 = H2
        E = 0
        for k in range(len(A_dense)):
            i = int(A_dense[k, 0])
            j = int(A_dense[k, 1])
            r_ij = A_dense[k, 2]
            e1_ij = r_ij - U1[i, :].dot(U2.T[:, j]) - H1[i, :].dot(H2.T[:, j])
            E += pow(e1_ij, 2)
            for l in range(r):
                _U1[i, l] = U1[i, l] + alpha * (2 * e1_ij * U2[j, l] - 2 * lambda_u * U1[i, l])
                _U2[j, l] = U2[j, l] + alpha * (2 * e1_ij * U1[i, l] - 2 * lambda_u * U2[j, l])
                #E += lambda_u * (pow(U1[i, k], 2) + pow(U2[j, k], 2))
            for l in range(r_):
                _H1[i, l] = H1[i, l] + alpha * (2 * e1_ij * H2[j, l] - 2 * lambda_h * H1[i, l])
                _H2[j, l] = H2[j, l] + alpha * (2 * e1_ij * H1[i, l] - 2 * lambda_h * H2[j, l])
                #E += lambda_h * (pow(H1[i, k], 2) + pow(H2[j, k], 2))
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i, j] > -1:
                    e2_ij = X[i, j] - U1[i, :].dot(V.T[:, j])
                    E += pow(e2_ij, 2)
                    for k in range(r):
                        _U1[i, k] = _U1[i, k] +alpha * (2 * e2_ij * V[j, k])
                        _V[j, k] = V[j, k] + alpha * (2 * e2_ij * U1[i, k] - 2 * lambda_v * V[j, k])
                        #E += lambda_v * (pow(V[j, k], 2))
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if Y[i, j] > 0:
                    e3_ij = Y[i, j] - U2[i, :].dot(V.T[:, j])
                    E += pow(e3_ij, 2)
                    for k in range(r):
                        _U2[i, k] = _U2[i, k] + alpha * (2 * e3_ij * V[j, k])
                        _V[j, k] = _V[j, k] + alpha * (2 * e3_ij * U2[i, k])

        e4 = lambda_u * (np.sum(U1 ** 2) + np.sum(U2 ** 2))
        e5 = lambda_h * (np.sum(H1 ** 2) + np.sum(H2 ** 2))
        e6 = lambda_v * (np.sum(V ** 2))
        #print ("E: " + str(E))
        E += e4 + e5 + e6
        #print (E)
        A_rmse = A_RMSE(valid_set, _U1, _U2, _H1, _H2)
        X_rmse = X_RMSE(X_valid, X, _U1, _V)
        Y_rmse = Y_RMSE(Y_valid, Y, _U2, _V)
        print("A_RMSE: " + str(A_rmse))
        print("X_RMSE: " + str(X_rmse))
        print("Y_RMSE: " + str(Y_rmse))
        print("--------------------")
        mean_rmse = (beta*A_rmse + 0.5*(1-beta)*X_rmse + 0.5*(1-beta)*Y_rmse)
        print("Mean_RMSE: " + str(mean_rmse))
        if prev_rmse < A_rmse and prev_mean_rmse < mean_rmse:
            break
        prev_mean_rmse = mean_rmse
        prev_rmse = A_rmse
        U1 = _U1
        U2 = _U2
        V = _V
        H1 = _H1
        H2 = _H2
    return [U1, U2, V, H1, H2]

import json
from math import exp
import numpy as np
import pandas as pd



def get_reviews(datafile):
    user_dict = {}
    item_dict = {}
    feature_dict ={}
    aspect_index = 0

    for tupel in datafile.iterrows():
        line = tupel[0]
        user_id = datafile["user_id"][line]
        item_id = datafile["business_id"][line]
        list_len = len(datafile['absa'][line])
        if user_id not in user_dict:
            user_dict[user_id] = []
        if item_id not in item_dict:
            item_dict[item_id] = []
        for i in range(0, list_len):
            feature = datafile['absa'][line][i]['aspect']
            polarity = datafile['absa'][line][i]['polarity']
            if feature not in feature_dict:
                feature_dict[feature] = aspect_index
                aspect_index = aspect_index+1
            user_dict[user_id].append([feature, polarity])
            item_dict[item_id].append([feature, polarity])
    return [feature_dict, user_dict, item_dict]


def get_index(user_dict, product_dict):
    user_index = {}
    product_index = {}
    index = 0
    for user in user_dict.keys():
        user_index[user] = index
        index += 1
    index = 0
    for product in product_dict.keys():
        product_index[product] = index
        index += 1
    return [user_index, product_index]


def get_user_item_matrix(datafile, user_index, product_index):
    num_users = len(user_index)
    num_items = len(product_index)
    result = np.zeros((num_users, num_items))
    num_reviews = len(datafile)
    result_dense = np.zeros((num_reviews, 3))
    for line in datafile.iterrows():
        i = line[0]
        user_id = datafile['user_id'][i]
        product_id = datafile['business_id'][i]
        user = user_index[user_id]
        product = product_index[product_id]
        rating = datafile['stars'][i]
        result[user, product] = rating
        result_dense[i, 0] = user
        result_dense[i, 1] = product
        result_dense[i, 2] = rating
    return result, result_dense

def split_by_time(datafile, portion):

    datafile = datafile.sort_values(['user_id', 'date'], ascending=[True, True])
    datafile = datafile.reset_index(drop=True)
    test = datafile[datafile['user_id']=='-1']
    cnt = 0
    for i in range(1, len(datafile)):
        cnt += 1
        if datafile['user_id'][i] != datafile['user_id'][i-1] or i == (len(datafile) - 1):
            j = round(cnt*portion)
            test = test.append(datafile.iloc[(i - int(j)):i])
            cnt = 0
    datafile = datafile.drop(datafile.index[test.index])
    datafile = datafile.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return test, datafile

def get_user_feature_matrix(user_dict, user_index, aspect_index, N):
    result = np.zeros((len(user_index), len(aspect_index)))
    for key in user_dict.keys():
        index_user = user_index[key]
        user_reviews = user_dict[key]
        count_dict = {}
        for review in user_reviews:
            feature = review[0]
            if feature not in aspect_index:
                continue
            aspect = aspect_index[feature]
            if aspect not in count_dict:
                count_dict[aspect] = 0;
            count_dict[aspect] += 1
        for aspect in count_dict.keys():
            count = count_dict[aspect]
            result[index_user, aspect] = 1 + (N - 1) * (2 / (1 + exp(-count)) - 1)
    return result


def get_product_feature_matrix(product_dict, product_index, aspect_index, N):
    result = np.zeros((len(product_index), len(aspect_index)))
    for key in product_dict.keys():
        index_product = product_index[key]
        product_reviews = product_dict[key]
        count_dict = {}
        for review in product_reviews:
            feature = review[0]
            polarity = review[1]
            if polarity == 'negative':
                s = -1
            elif polarity == 'positive':
                s = 1
            elif polarity == 'neutral':
                s = 0.5
            aspect = aspect_index[feature]
            if aspect not in count_dict:
                count_dict[aspect] = [];
            count_dict[aspect].append(s)
        for aspect in count_dict.keys():
            count = sum(count_dict[aspect])
            result[index_product, aspect] = 1 + (N - 1) / (1 + exp(-count))
    return result

def get_X_validation(X):
    X_valid = np.copy(X)
    mask = np.random.choice([0, 1], size=X.shape, p=[0.8, 0.2] ).astype(np.bool)
    X[mask] = -1
    X_valid[~mask] = -1
    return X, X_valid
#import get_matrices
#import train

import numpy as np
import pandas as pd
import pickle

def get_user_care(user_id, user_index, aspect_index, user_feature_matrix):
    user_i = user_index[user_id]
    for k, v in aspect_index.items():
        print (k + " : " + str(user_feature_matrix[user_i, v]))

def get_item_care(product_id, product_index, aspect_index, product_feature_matrix):
    user_i = product_index[product_id]
    for k, v in aspect_index.items():
        print (k + " : " + str(product_feature_matrix[user_i, v]))


def top_k(user_i, X, Y, A, A_test, user_index, user_feature_matrix, k, alpha):
    user_care = user_feature_matrix[user_i, :]
    idx = np.argpartition(user_care, -k)
    idx = idx[-k:]
    R_i = np.zeros(Y.shape[0])
    value_counts = pd.value_counts(test.user_id)

    for i in range(R_i.shape[0]):
        # maybe change X to user_feature_matrix and Y_ to real Y
        if A_test[user_i, i] != 0: # only if not in train and was rated in the test set!
            tmp = X[user_i, idx].dot(Y[i, idx].T) / k / 5.0
            R_i[i] = tmp * alpha + (1 - alpha) * A[user_i, i]
        else:
            R_i[i] = 0
    idx = R_i.argsort()[-5:][::-1]
    #item_id = get_item_id(product_index, idx)
    return idx

def DCG(items_list):
    dcg = 0
    for i in range(len(items_list)):
        dcg += (2**items_list[i] - 1)/(np.log10(i+2))
    return dcg


def ndcg(A_test_dense,  X, Y, A, A_test, user_index, user_feature_matrix, k, alpha):
    top_k_rec = top_true = np.zeros(5)
    dcg = 0
    idcg = 0
    Sum_NDCG = 0
    cnt = 0
    for i in range(0, int(A_test_dense[(len(A_test_dense) - 1), 0])):
        curr_user_arr = np.where(A_test_dense[:, 0] == i)[0]
        if len(curr_user_arr) > 10:
            cnt += 1
            top_k_rec = top_k(i, X, Y, A, A_test, user_index, user_feature_matrix, k, alpha)
            # top_k_ratings = np.zeros(len(top_k))
            top_k_ratings = A_test[i, top_k_rec].astype(int)
            dcg = DCG(top_k_ratings)
            idx = A_test[i, :].argsort()[-5:][::-1]
            top_true = A_test[i, idx].astype(int)
            idcg = DCG(np.sort(top_true)[::-1])
            ndcg = dcg/idcg
            Sum_NDCG += ndcg
    return Sum_NDCG/cnt

def get_item_id(product_index, index):
    top = [0]*5
    for k, v in product_index.items():
        if v == index[0]:
           top[0] = k
        elif v == index[1]:
            top[1] = k
        elif v == index[2]:
            top[2] = k
    return top



if __name__ == "__main__":

    data_frame_path = "../data/Toronto_rest_reviews_absa_final.pkl"
    datafile = pd.read_pickle(data_frame_path)
    datafile['date'] = pd.to_datetime(datafile['date'])
    [test, train] = y_get_matrices.split_by_time(datafile, 0.2)
    [feature_index, user_dict, product_dict] = y_get_matrices.get_reviews(train)
    [validation, train] = y_get_matrices.split_by_time(train, 0.2)
    [user_index, product_index] = y_get_matrices.get_index(user_dict, product_dict)
    A_train, A_train_dense = y_get_matrices.get_user_item_matrix(train, user_index, product_index)
    A_test, A_test_dense = y_get_matrices.get_user_item_matrix(test, user_index, product_index)
    A_valid, A_valid_dense = y_get_matrices.get_user_item_matrix(validation, user_index, product_index)

    # with open('/home/ise/yuval/RS/project/gitHubCode/Explainable-Recommendation-master/data/melaph1.pickle', 'wb') as f:
    #     pickle.dump([A_test, A_test_dense, A_train, A_train_dense,A_valid, A_valid_dense, test, train, validation, user_index, feature_index, user_dict, product_index, product_dict], f)

    # with open('/home/ise/yuval/RS/project/gitHubCode/Explainable-Recommendation-master/data/melaph1.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    #    A_test, A_test_dense, A_train, A_train_dense, A_valid, A_valid_dense, test, train, validation, user_index, feature_index, user_dict, product_index, product_dict = pickle.load(f)

    user_feature_matrix = y_get_matrices.get_user_feature_matrix(user_dict, user_index, feature_index, 5)
    product_feature_matrix = y_get_matrices.get_product_feature_matrix(product_dict, product_index, feature_index, 5)

    print("start training")
    #[U1, U2, V, H1, H2] = y_train.start_training(A_train_dense, A_train, user_feature_matrix, product_feature_matrix, A_valid_dense, 50, 50, 0.01, 0.01, 0.01, 0.01, 0.01, 400)
    [U1, U2, V, H1, H2] = y_train.training(A_train_dense, A_train, user_feature_matrix, product_feature_matrix, A_valid_dense, 25, 25, 0.01, 0.01, 0.01, 0.01, 0.01, 200, 0.002, 0.4)
    X_ = U1.dot(V.T)
    Y_ = U2.dot(V.T)
    A_ = U1.dot(U2.T) + H1.dot(H2.T)

    # with open('/home/ise/yuval/RS/project/gitHubCode/Explainable-Recommendation-master/data/results.pickle', 'wb') as f:
    #     pickle.dump([U1, U2, V, H1, H2, X_, Y_, A_], f)

    # with open('/home/ise/yuval/RS/project/gitHubCode/Explainable-Recommendation-master/data/results.pickle', 'rb') as f:
    #     U1, U2, V, H1, H2, X_, Y_, A_ = pickle.load(f)

    print("Test RMSE: " +str(y_train.A_RMSE(A_test_dense, U1, U2, H1, H2)))
    ndcg = ndcg(A_test_dense, X_, Y_, A_, A_test, user_index, user_feature_matrix, 3, 0.12)
    print("NDCG: " + str(ndcg))