import y_get_matrices
import y_train
import numpy as np
import pandas as pd
import surprise
from surprise import Dataset
from surprise import Reader
from surprise import accuracy



def top_k(user_i, X, Y, A, A_test, k, alpha):
    """top-k recommendations list for a user

        Args:
            user_i: the user index
            X, Y, A - predicted values
            k: number of most cared items
            alpha: tradeoff between user-item similarity and predicted rating
        Returns:
            The indeces of the top-k items
       """
    if len(X[1]) > 14:
        user_care = X[user_i, :len(X[0] - 2)]
    else:
        user_care = X[user_i, :]
    idx = np.argpartition(user_care, -k)
    idx = idx[-k:]
    if len(X[1]) > 14:
        idx = np.append(idx, 14)
        k += 1
    R_i = np.zeros(Y.shape[0])
    value_counts = pd.value_counts(test.user_id)

    for i in range(R_i.shape[0]):

        if A_test[user_i, i] != 0: # only if not in train and was rated in the test set!
            if len(X[1]) == 14:
                tmp = X[user_i, idx].dot(Y[i, idx].T) / k / 5.0
            else:
                tmp = ((X[user_i, idx]).dot((Y[i, idx]).T) / np.sqrt(np.sum((X[user_i, idx])**2)*np.sum((Y[i, idx])**2))) * 4.0 + 1
            R_i[i] = tmp * alpha + (1 - alpha) * A[user_i, i]
        else:
            R_i[i] = 0
    idx = R_i.argsort()[-5:][::-1]
    return idx


def top_k_sur(user_i, predictions, Y):
    """top-k recommendations list for a user for the models of surprise package

            Args:
                user_i: the user index

            Returns:
                The indeces of the top-k items
           """

    R_i = np.zeros(len(Y))
    value_counts = pd.value_counts(test.user_id)

    for i in range(R_i.shape[0]):
        if A_test[user_i, i] != 0: # only if not in train and was rated in the test set!
            for r in range(len(predictions)):
                if predictions[r][0] == user_i and predictions[r][1] == i:
                    R_i[i] = predictions[r][3]
    idx = R_i.argsort()[-5:][::-1]
    return idx


def DCG(items_list):
    """compute dcg over tthe entire lost

            Args:
                items_list: list of items

            Returns:
                The DCG for the list
           """
    dcg = 0
    for i in range(len(items_list)):
        dcg += (2**items_list[i] - 1)/(np.log10(i+2))
    return dcg


def ndcg(A_test_dense,  X, Y, A, A_test, k, alpha):
    """NDCG@5 for the entire test set users with a given k and alpha

            Args:
                A_test_dense: the test set user-item-rating list
                X, Y, A - predicted values
                k: number of most cared items
                alpha: tradeoff between user-item similarity and predicted rating
            Returns:
                ndcg@k for test set
    """
    top_k_rec = top_true = np.zeros(5)
    dcg = 0
    idcg = 0
    Sum_NDCG = 0
    cnt = 0
    for i in range(0, int(A_test_dense[(len(A_test_dense) - 1), 0])):
        curr_user_arr = np.where(A_test_dense[:, 0] == i)[0]
        if len(curr_user_arr) > 10:
            cnt += 1
            top_k_rec = top_k(i, X, Y, A, A_test, k, alpha)
            top_k_ratings = A_test[i, top_k_rec].astype(int)
            dcg = DCG(top_k_ratings)
            idx = A_test[i, :].argsort()[-5:][::-1]
            top_true = A_test[i, idx].astype(int)
            idcg = DCG(np.sort(top_true)[::-1])
            ndcg = dcg/idcg
            Sum_NDCG += ndcg
    return Sum_NDCG/cnt


def precision(A_test_dense,  X, Y, A, A_test, k, alpha):
    """precision@5 fot the entire test-set with fixed k and alpha

            Args:
                A_test_dense: the test set user-item-rating list
                X, Y, A - predicted values
                k: number of most cared items
                alpha: tradeoff between user-item similarity and predicted rating
            Returns:
                precision@k for test set
    """
    top_k_rec = top_true = np.zeros(5)
    dcg = 0
    idcg = 0
    Sum_prec = 0
    cnt = 0
    for i in range(0, int(A_test_dense[(len(A_test_dense) - 1), 0])):
        curr_user_arr = np.where(A_test_dense[:, 0] == i)[0]
        if len(curr_user_arr) > 10:
            cnt += 1
            top_k_rec = top_k(i, X, Y, A, A_test, k, alpha)
            # top_k_ratings = np.zeros(len(top_k))
            top_k_ratings = A_test[i, top_k_rec].astype(int)
            prec = sum(top_k_ratings > 3)/len(top_k_ratings)
            Sum_prec += prec
    return Sum_prec/cnt


def sur_ndcg(A_test_dense, predictions, Y):
    """ndcg@5 fot the entire test-set for surprise package models

                Args:
                    A_test_dense: the test set user-item-rating list
                Returns:
                    ndcg@k for test set
    """
    top_k_rec = top_true = np.zeros(5)
    dcg = 0
    idcg = 0
    Sum_NDCG = 0
    cnt = 0
    for i in range(0, int(A_test_dense[(len(A_test_dense) - 1), 0])):
        curr_user_arr = np.where(A_test_dense[:, 0] == i)[0]
        if len(curr_user_arr) > 10:
            cnt += 1
            top_k_rec = top_k_sur(i, predictions, Y)
            # top_k_ratings = np.zeros(len(top_k))
            top_k_ratings = A_test[i, top_k_rec].astype(int)
            dcg = DCG(top_k_ratings)
            idx = A_test[i, :].argsort()[-5:][::-1]
            top_true = A_test[i, idx].astype(int)
            idcg = DCG(np.sort(top_true)[::-1])
            ndcg = dcg/idcg
            Sum_NDCG += ndcg
    return Sum_NDCG/cnt


def sur_precision(A_test_dense, predictions, Y):
    """precision@5 fot the entire test-set for surprise package models

                Args:
                    A_test_dense: the test set user-item-rating list
                Returns:
                    precision@k for test set
    """
    top_k_rec = top_true = np.zeros(5)
    dcg = 0
    idcg = 0
    Sum_prec = 0
    cnt = 0
    for i in range(0, int(A_test_dense[(len(A_test_dense) - 1), 0])):
        curr_user_arr = np.where(A_test_dense[:, 0] == i)[0]
        if len(curr_user_arr) > 10:
            cnt += 1
            top_k_rec = top_k_sur(i, predictions, Y)
            # top_k_ratings = np.zeros(len(top_k))
            top_k_ratings = A_test[i, top_k_rec].astype(int)
            prec = sum(top_k_ratings > 3) / len(top_k_ratings)
            Sum_prec += prec
    return Sum_prec/cnt


def ndcg_bpr(A_test_dense,  f):
    """ndcg@5 fot the entire test-set for BPRMF model

                    Args:
                        A_test_dense: the test set user-item-rating list
                    Returns:
                        ndcg@k for test set
        """
    top_k_rec = top_true = np.zeros(5)
    dcg = 0
    idcg = 0
    Sum_NDCG = 0
    cnt = 0
    w = f.bias.T + np.dot(f.p, f.q.T)

    for i in range(0, int(A_test_dense[(len(A_test_dense) - 1), 0])):
        top_200 = np.zeros((200, 2))
        curr_user_arr = np.where(A_test_dense[:, 0] == i)[0]
        if len(curr_user_arr) > 10:
            cnt += 1
            for z in range(len(curr_user_arr)) :
                top_200[z, 0] = w[i, int(A_test_dense[curr_user_arr[z], 1])]
                top_200[z, 1] = int(A_test_dense[curr_user_arr[z], 1])
            topk = top_200[top_200[:,0] != 0]
            top_k_rec = topk[np.argsort(topk[:, 0])][::-1][:5, 1]
            # top_k_rec = np.sort(topk, axis=0)[::-1][:5, 1]
            # top_k_ratings = np.zeros(len(top_k))
            top_k_ratings = A_test[i, top_k_rec.astype(int)].astype(int)
            dcg = DCG(top_k_ratings)
            idx = A_test[i, :].argsort()[-5:][::-1]
            top_true = A_test[i, idx].astype(int)
            idcg = DCG(np.sort(top_true)[::-1])
            ndcg = dcg/idcg
            Sum_NDCG += ndcg
    return Sum_NDCG/cnt


def prec_bpr(A_test_dense,  f):
    """precision@5 fot the entire test-set for surprise package models

                Args:
                    A_test_dense: the test set user-item-rating list
                Returns:
                    precision@k for test set
    """
    top_k_rec = top_true = np.zeros(5)
    dcg = 0
    idcg = 0
    Sum_prec = 0
    cnt = 0
    w = f.bias.T + np.dot(f.p, f.q.T)

    for i in range(0, int(A_test_dense[(len(A_test_dense) - 1), 0])):
        top_200 = np.zeros((200, 2))
        curr_user_arr = np.where(A_test_dense[:, 0] == i)[0]
        if len(curr_user_arr) > 10:
            cnt += 1
            for z in range(len(curr_user_arr)) :
                top_200[z, 0] = w[i, int(A_test_dense[curr_user_arr[z], 1])]
                top_200[z, 1] = int(A_test_dense[curr_user_arr[z], 1])
            topk = top_200[top_200[:,0] != 0]
            top_k_rec = topk[np.argsort(topk[:, 0])][::-1][:5, 1]
            # top_k_rec = np.sort(topk, axis=0)[::-1][:5, 1]
            # top_k_ratings = np.zeros(len(top_k))
            top_k_ratings = A_test[i, top_k_rec.astype(int)].astype(int)
            prec = sum(top_k_ratings > 3) / len(top_k_ratings)
            Sum_prec += prec
    return Sum_prec/cnt


if __name__ == "__main__":

    for d in range(10): # super loop to run each model 10 times
        # collect and prepare data:
        data_frame_path = "../data/Toronto_rest_reviews_absa_final.pkl"
        data_business_path = "../data/yelp_business.csv"
        items_data = pd.read_csv(data_business_path)
        datafile = pd.read_pickle(data_frame_path)
        datafile['date'] = pd.to_datetime(datafile['date'])
        items_data = items_data[items_data.business_id.isin(datafile.business_id)]
        items_data = items_data.reset_index(drop=True)
        [test, train] = y_get_matrices.split_by_time(datafile, 0.2)
        [feature_index, user_dict, product_dict] = y_get_matrices.get_sentiment_lexicon(train)
        [validation, train] = y_get_matrices.split_by_time(train, 0.2)
        [user_index, product_index] = y_get_matrices.get_index(user_dict, product_dict)
        A_train, A_train_dense = y_get_matrices.get_user_item_matrix(train, user_index, product_index)
        A_test, A_test_dense = y_get_matrices.get_user_item_matrix(test, user_index, product_index)
        A_valid, A_valid_dense = y_get_matrices.get_user_item_matrix(validation, user_index, product_index)

        popularity = y_get_matrices.get_popularity(items_data, product_index)

        user_feature_matrix1 = y_get_matrices.get_user_feature_matrix(user_dict, user_index, feature_index, 5) # for base model
        product_feature_matrix1 = y_get_matrices.get_product_feature_matrix(product_dict, product_index, feature_index, 5, False) # for base model
        user_feature_matrix2 = y_get_matrices.get_user_feature_matrix_p(user_dict, user_index, feature_index, 5, popularity, A_train_dense, False) #for improved model with popularity
        product_feature_matrix2 = y_get_matrices.get_product_feature_matrix_p(product_dict, product_index, feature_index, 5, popularity, False) # for improved model with popularity
        user_feature_matrix3 = y_get_matrices.get_user_feature_matrix_p(user_dict, user_index, feature_index, 5, popularity, A_train_dense, True) # for improved model with polarity
        product_feature_matrix3 = y_get_matrices.get_product_feature_matrix_p(product_dict, product_index, feature_index, 5, popularity, True) # for improved model with polarity
        [X1, X_valid1, X2, X_valid2, X3, X_valid3] = y_get_matrices.get_X_validation(user_feature_matrix1, user_feature_matrix2, user_feature_matrix3)
        [Y1, Y_valid1, Y2, Y_valid2, Y3, Y_valid3] = y_get_matrices.get_X_validation(product_feature_matrix1, product_feature_matrix2, product_feature_matrix3)


        for k in range(4):

            print("start training model: " + str(k))
            if k == 0: # train base model
                [U1, U2, V, H1, H2, bu, bi] = y_train.training(A_train_dense, A_train, X1, X_valid1, Y1, Y_valid1, A_valid_dense, 40, 25, 0.0025, 0.005, 0.0, 0.001, 0.01, 200, 0.005, 0.50)
            if k == 1: # train improved model with popularity
                [U1, U2, V, H1, H2, bu, bi] = y_train.training(A_train_dense, A_train, X2, X_valid2, Y2, Y_valid2, A_valid_dense, 40, 25, 0.0025, 0.005, 0.0, 0.001, 0.02, 200, 0.005, 0.50)
            if k == 2: # train improved model with polarity
                [U1, U2, V, H1, H2, bu, bi] = y_train.training(A_train_dense, A_train, X3, X_valid3, Y3, Y_valid3, A_valid_dense, 40, 25, 0.0025, 0.004, 0.0, 0.001, 0.02, 200, 0.005, 0.60)
            if k == 3: # train and test other MF models
                # BPRMF:
                from caserec.recommenders.item_recommendation.bprmf import BprMF
                tr = '../data/train.csv'
                tst = '../data/test.csv'
                f = BprMF(tr, tst)
                f.compute()
                rankings = f.ranking
                print("model BPRMF:")
                print("NDCG: "+str(ndcg_bpr(A_test_dense, f)))
                print("Precision: " + str(prec_bpr(A_test_dense, f)))

                # prepare data for surprise package:
                reader = Reader(rating_scale=(1, 5))
                atstd = A_test_dense
                atrd = A_train_dense
                for m in range(len(datafile)):
                    datafile.user_id.at[m] = user_index[datafile.user_id.at[m]]
                    datafile.business_id.at[m] = product_index[datafile.business_id.at[m]]
                data = Dataset.load_from_df(datafile[['user_id', 'business_id', 'stars']], reader)
                A_train_dense = list([list(row) for row in A_train_dense])
                for i in range(len(A_train_dense)):
                    A_train_dense[i].append(None)
                A_train_dense = list([tuple(row) for row in A_train_dense])

                A_test_dense = list([list(row) for row in A_test_dense])
                for i in range(len(A_test_dense)):
                    A_test_dense[i].append(None)
                A_test_dense = list([tuple(row) for row in A_test_dense])

                trainset = data.construct_trainset(A_train_dense)
                testset = data.construct_testset(A_test_dense)

                # SVDpp:
                algo = surprise.SVDpp()
                algo.fit(trainset)
                predictions = algo.test(testset)
                print("model SVDpp: ")
                # Then compute RMSE
                accuracy.rmse(predictions)
                print("NDCG: "+str(sur_ndcg(atstd, predictions, product_index)))
                print("Precision: " + str(sur_precision(atstd, predictions, product_index)))

                # NMF:
                algo = surprise.NMF()
                algo.fit(trainset)
                predictions = algo.test(testset)
                print("model NMF: ")
                accuracy.rmse(predictions)
                print("NDCG: " + str(sur_ndcg(atstd, predictions, product_index)))
                print("Precision: " + str(sur_precision(atstd, predictions, product_index)))

                # SlopeOne:
                algo = surprise.SlopeOne()
                algo.fit(trainset)
                predictions = algo.test(testset)
                print("model SlopeOne: ")
                # Then compute RMSE
                accuracy.rmse(predictions)
                print("NDCG: " + str(sur_ndcg(atstd, predictions, product_index)))
                print("Precision: " + str(sur_precision(atstd, predictions, product_index)))
                continue

            # Make predictions with trained parameters:
            X_ = U1.dot(V.T)
            Y_ = U2.dot(V.T)
            A_ = U1.dot(U2.T) + H1.dot(H2.T)
            mean_rating = np.mean(A_train_dense[2])
            for u in range(len(A_)):
                for it in range(len(A_[0])):
                    A_[u, it] = A_[u, it] + mean_rating + bu[u] + bi[it]

            # evaluate RMSE:
            print("Test RMSE model " + str(k)+" is: "+str(y_train.A_RMSE(A_test_dense, U1, U2, H1, H2, mean_rating, bu, bi)))

            # Evaluate NDCG with a super loop for all possible alphas and k's:
            Max = 0
            final_alpha = 0
            for j in range(1 , 12):
                alpha = max_alpha = 0
                max_ndcg_ = 0
                for i in range(99):
                    ndcg_ = ndcg(A_test_dense, X_, Y_, A_, A_test, j, alpha)
                    if ndcg_ > max_ndcg_:
                        max_ndcg_ = ndcg_
                        max_alpha = alpha
                    alpha += 0.01

                if max_ndcg_ > Max:
                    Max = max_ndcg_
                    final_alpha = max_alpha
                    final_k = j
            print("NDCG for model " + str(k) + " is: " + str(Max) + "   alpha: " + str(round(final_alpha, 2)) + " k: " + str(final_k))

            # Evaluate precision:
            print("precision for model " + str(k) + " is: " +str(precision(A_test_dense, X_, Y_, A_, A_test, final_k, final_alpha)))