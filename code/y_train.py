import y_get_matrices
import numpy as np
import pandas as pd
import pickle



def A_RMSE(A_dense, U1, U2, H1, H2, mean_rating, bu, bi):
    """RMSE for A validation set user-item-ratings list

            Args:
                parameters for predicting ratings
            Returns:
                RMSE
    """
    E = 0
    for k in range(len(A_dense)):
        i = int(A_dense[k, 0])
        j = int(A_dense[k, 1])
        r_ij = A_dense[k, 2]
        e = r_ij - (U1[i, :].dot(U2.T[:, j]) + H1[i, :].dot(H2.T[:, j]) +mean_rating + bu[i] + bi[j])
        E += e**2

    return (E/len(A_dense))**0.5


def X_RMSE(X_valid, X, U1, V):
    """RMSE for X validation sets list

                        Args:
                            parameters for predicting ratings
                        Returns:
                            RMSE
    """
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
    """RMSE for Y validation sets list

                Args:
                    parameters for predicting ratings
                Returns:
                    RMSE
    """
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
    """Training with the EFM suggested optimization algorithm - bad results!
                    Args:
                        learning hyper-parameters r, r_, lambda_x, lambda_y, lambda_u, lambda_h, lambda_v
                        True values A, X, Y
                    Returns:
                        latent factors matrices U1, U2, V, H1, H2
    """
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

def training(A_dense, A, X, X_valid, Y, Y_valid, valid_set,  r, r_, gamma_x, gamma_y, lambda_y, lambda_x, lambda_a, T, gamma_a, beta):
    """Training with the EFM+ suggested optimization algorithm
                Args:
                    r: num of latent factors
                    r_: num of latent factors
                    gammas: learning rates
                    lambdas: regularization coefficients
                    True values A, X, Y
                    validation sets X_valid, Y_valid, valid_set
                    beta: hyper parameter controlling the convergence criteria
                Returns:
                    latent factors matrices U1, U2, V, H1, H2
    """
    m = X.shape[0]
    p = X.shape[1]
    n = Y.shape[0]
    # initialize parameters randomly:
    U1 = np.round(np.random.uniform(-0.05, 0.05, (m, r)), 4)
    U2 = np.round(np.random.uniform(-0.05, 0.05, (n, r)), 4)
    V = np.round(np.random.uniform(-0.05, 0.05, (p, r)), 4)
    H1 = np.round(np.random.uniform(-0.05, 0.05, (m, r_)), 4)
    H2 = np.round(np.random.uniform(-0.05, 0.05, (n, r_)), 4)
    bu = np.round(np.random.uniform(-0.05, 0.05, (len(X))), 4) # adding bias
    bi = np.round(np.random.uniform(-0.05, 0.05, (len(Y))), 4) # adding bias
    mean_rating = np.mean(A_dense[2])
    t = 0
    prev_rmse = 100
    while t < T: # run on max num of iterations
        t += 1
        _U1 = U1
        _U2 = U2
        _V = V
        _H1 = H1
        _H2 = H2
        E = 0

        # optimize with respect to matrix A (ratings matrix)
        for k in range(len(A_dense)):
            i = int(A_dense[k, 0])
            j = int(A_dense[k, 1])
            r_ij = A_dense[k, 2]
            e1_ij = r_ij - (U1[i, :].dot(U2.T[:, j]) + H1[i, :].dot(H2.T[:, j])+ mean_rating +bu[i] +bi[j])
            E += pow(e1_ij, 2)
            for l in range(r):
                _U1[i, l] = U1[i, l] + gamma_a * ( e1_ij * U2[j, l] - lambda_a * U1[i, l])
                _U2[j, l] = U2[j, l] + gamma_a * ( e1_ij * U1[i, l] - lambda_a * U2[j, l])

            for l in range(r_):
                _H1[i, l] = H1[i, l] + gamma_a * ( e1_ij * H2[j, l] - lambda_a * H1[i, l])
                _H2[j, l] = H2[j, l] + gamma_a * ( e1_ij * H1[i, l] - lambda_a * H2[j, l])

            bu[i] += (gamma_a*(e1_ij - lambda_a*bu[i]))
            bi[j] += (gamma_a * (e1_ij - lambda_a * bi[j]))

        # optimize with respect to X
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i, j] > -1:
                    e2_ij = X[i, j] - U1[i, :].dot(V.T[:, j])
                    E += pow(e2_ij, 2)
                    for k in range(r):
                        _U1[i, k] = _U1[i, k] + gamma_x * (e2_ij * V[j, k] - lambda_y*_U1[i, k])
                        _V[j, k] = V[j, k] + gamma_x * (e2_ij * U1[i, k] - lambda_x * V[j, k])

        # optimize with respect to Y
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if Y[i, j] > 0:
                    e3_ij = Y[i, j] - U2[i, :].dot(V.T[:, j])

                    for k in range(r):
                        _U2[i, k] = _U2[i, k] + gamma_y* (e3_ij * V[j, k] - lambda_y*_U2[i, k])
                        _V[j, k] = _V[j, k] + gamma_y * (e3_ij * U2[i, k] - lambda_y * V[j, k])



        A_rmse = A_RMSE(valid_set, _U1, _U2, _H1, _H2, mean_rating, bu, bi)
        X_rmse = X_RMSE(X_valid, X, _U1, _V)
        Y_rmse = Y_RMSE(Y_valid, Y, _U2, _V)

        print("A_RMSE: " + str(A_rmse))
        print("X_RMSE: " + str(X_rmse))
        print("Y_RMSE: " + str(Y_rmse))
        print("--------------------")

        mean_rmse = (beta*A_rmse + 0.5*(1-beta)*X_rmse + 0.5*(1-beta)*Y_rmse) # for convergence criteria
        print("Mean_RMSE: " + str(mean_rmse))

        # break if convergence criteria is met
        if prev_rmse < A_rmse and prev_mean_rmse < mean_rmse:
            break
        prev_mean_rmse = mean_rmse
        prev_rmse = A_rmse
        U1 = _U1
        U2 = _U2
        V = _V
        H1 = _H1
        H2 = _H2

    return [U1, U2, V, H1, H2, bu, bi]