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