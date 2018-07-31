import json
from math import exp
import numpy as np
import pandas as pd



def get_sentiment_lexicon(datafile):
    """get the sentiment lexicon out of the datafile

                Args:
                    datafile: table with all relevant data for the project including reviews and ABSA
                Returns:
                    features (aspects) dictionary, user-review dict that holds sentiment lexicon entries
                    of the user and item-review dict that holds sentiment lexicon entries
                    of the item
    """
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
            fetaure_confidence = datafile['absa'][line][i]['aspect_confidence']
            polarity = datafile['absa'][line][i]['polarity']
            polarity_confidence = datafile['absa'][line][i]['polarity_confidence']
            if feature not in feature_dict:
                feature_dict[feature] = aspect_index
                aspect_index = aspect_index+1
            user_dict[user_id].append([feature, fetaure_confidence, polarity, polarity_confidence])
            item_dict[item_id].append([feature, fetaure_confidence, polarity, polarity_confidence])
    return [feature_dict, user_dict, item_dict]


def get_index(user_dict, product_dict):
    """get the users and items indexes dicts

    """
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
    """get user-item rating matrix

                    Args:
                        datafile: table with all relevant data for the project including reviews and ABSA
                    Returns:
                        The sparse rating matrix and a dense representation
        """
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
    """split the data according to the time-stamp, by a given
       proportion. the split is done in respect to each user entries

                    Args:
                        datafile: table with all relevant data for the project including reviews and ABSA
                        portion: split ratio
                    Returns:
                        test set, train set
        """
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
    """get the user-feature attention matrix X of the base model EFM+
                    Args:
                        N: max rating
                    Returns:
                        matrix X
    """
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


def get_user_feature_matrix_p(user_dict, user_index, aspect_index, N, popularity, A_dense, Polarity):
    """get the user-feature attention matrix X of the improved model with popularity and polarity range
                Args:
                    N: max rating
                Returns:
                    matrix X
    """
    result = np.zeros((len(user_index), len(aspect_index)))
    for key in user_dict.keys():
        index_user = user_index[key]
        user_reviews = user_dict[key]
        count_dict = {}
        max = 0
        min = 1000
        for review in user_reviews:
            feature = review[0]
            if feature not in aspect_index:
                continue
            aspect = aspect_index[feature]
            if aspect not in count_dict:
                count_dict[aspect] = 0;
            if Polarity == False:
                count_dict[aspect] += 1
            else:
                count_dict[aspect] += review[1]
        for aspect in count_dict.keys():
            count = count_dict[aspect]
            if count > max:
                max = count
            if count < min:
                min = count
        for aspect in count_dict.keys():
            count = count_dict[aspect]
            result[index_user, aspect] = (((count - min)/(max - min))*5)

    if len(popularity) > 0:
        col = np.zeros((len(result), 1))
        result = np.append(result, col, axis=1)
        for i in range(len(result)):
            items = A_dense[A_dense[:, 0] == i][:, 1]
            items = items.astype(int)
            result[i, len(result[1]) - 1] = np.mean(popularity[items, 1])
    return result


def get_product_feature_matrix(product_dict, product_index, aspect_index, N, Pol):
    """get the item-feature quality matrix Y of the base model or improved with polarity range
                    Args:
                        N: max rating
                    Returns:
                        matrix Y
    """
    result = np.zeros((len(product_index), len(aspect_index)))
    for key in product_dict.keys():
        index_product = product_index[key]
        product_reviews = product_dict[key]
        count_dict = {}
        maxC = 0
        minC = 100
        for review in product_reviews:
            feature = review[0]
            polarity = review[2]
            if polarity == 'negative':
                s = -1
            elif polarity == 'positive':
                s = 1
            elif polarity == 'neutral':
                s = 0.5
            if Pol:
                s = s*review[1]*review[3]
            aspect = aspect_index[feature]
            if aspect not in count_dict:
                count_dict[aspect] = [];
            count_dict[aspect].append(s)
        for aspect in count_dict.keys():
            count = sum(count_dict[aspect])
            if count/len(count_dict[aspect]) > maxC: maxC = count/len(count_dict[aspect])
            if count / len(count_dict[aspect]) < minC: minC = count / len(count_dict[aspect])
            result[index_product, aspect] = 1 + (N - 1) / (1 + exp(-count))

        if Pol: # compute in case asked - polarity range
            for key in product_dict.keys():
                index_product = product_index[key]
                product_reviews = product_dict[key]
                count_dict = {}
                max = 0
                for review in product_reviews:
                    feature = review[0]
                    polarity = review[2]
                    if polarity == 'negative':
                        s = -1
                    elif polarity == 'positive':
                        s = 1
                    elif polarity == 'neutral':
                        s = 0.5
                    if Pol:
                        s = s * review[1] * review[3]
                    aspect = aspect_index[feature]
                    if aspect not in count_dict:
                        count_dict[aspect] = [];
                    count_dict[aspect].append(s)
                for aspect in count_dict.keys():
                    count = sum(count_dict[aspect])
                    result[index_product, aspect] = (((count/len(count_dict[aspect]) - minC)/(maxC - minC))*5)
    return result

def get_product_feature_matrix_p(product_dict, product_index, aspect_index, N, popularity, Pol):
    """get the item-feature quality matrix Y of the improved with popularity model
                        Args:
                            N: max rating
                        Returns:
                            matrix Y
        """
    result = get_product_feature_matrix(product_dict, product_index, aspect_index, N, Pol)
    if len(popularity) > 0:
        popularity = np.reshape(popularity[:, 1], (len(popularity), 1))
        result = np.append(result, popularity, axis=1)
    return result


def get_X_validation(X1, X2, X3):
    """returns train and validation sets of X or Y matrices for all 3 model types
    """
    X_valid1 = np.copy(X1)
    X_valid2 = np.copy(X2)
    X_valid3 = np.copy(X3)
    mask = np.random.choice([0, 1], size=X2.shape, p=[0.8, 0.2] ).astype(np.bool)
    X2[mask] = -1
    X3[mask] = -1
    X_valid2[~mask] = -1
    X_valid3[~mask] = -1
    for i in range(len(X1[0])):
        for j in range(len(X1[1])):
            if X2[i, j] == -1:
                X1[i, j] = -1
            else:
                X_valid1[i, j] = -1
    return X1, X_valid1, X2, X_valid2, X3, X_valid3


def get_popularity(rest_data, item_dict):
    """get the the popularity aspect column vector for matrix Y
                Args:
                    rest_data: containing additional data onthe restaurants
                Returns:
                    matrix Y
    """
    max_review_count = rest_data.review_count.max()
    min_review_count = rest_data.review_count.min()
    result = np.zeros((len(rest_data), 2))
    for i in range(len(rest_data)):
        result[i, 0] = item_dict[rest_data.business_id[i]]
        result[i, 1] = (((rest_data.review_count[i] - min_review_count)/(max_review_count - min_review_count))*4 + 1)
    result = result[result[:, 0].argsort()]
    return result
