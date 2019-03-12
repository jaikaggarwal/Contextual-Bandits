import pandas as pd
import numpy.random as nprnd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from scipy.stats import invgamma
from scipy.stats import spearmanr

def posterior(y, X, m_pre, V_pre, a_pre, b_pre):
    """
    Want to the find the parameters of the posterior distribution given the equations in
    Linear Contextual Bandits.ipynb.
    :param y: data found
    :param X: design matrix of exploratory variables given
    :param m_pre: mean of the weights before update
    :param V_pre: covariance matrix of the weights before update
    :param a_pre: shape parameter before update
    :param b_pre: scale parameter before update
    :return:
    """
    # Set up matrices that are reused throughout
    size = len(y)
    X_transpose = np.transpose(X)
    V_inverted = np.invert(V_pre)
    residual = np.subtract(y, np.dot(X, m_pre))
    residual_transpose = np.transpose(residual)

    # Find parameters for weights
    V_post = np.invert(np.add(V_inverted, np.dot(X_transpose, X)))
    m_post_second_matrix = np.add(np.dot(V_inverted, m_pre), np.dot(X_transpose, y))
    m_post = np.dot(V_post, m_post_second_matrix)

    # Find parameters for variance parametrized by a and b
    a_post = a_pre + (size / 2)
    middle_val = np.invert(np.add(np.identity(size), np.dot(np.dot(X, V_pre), X_transpose)))
    b_post = b_pre + 0.5 * np.dot(np.dot(residual_transpose, middle_val), residual)

    # Find values for the variance and the weights
    var_from_invgamma = invgamma.rvs(a_post, 0, b_post, size=1)
    beta_draw = np.random.multivariate_normal(m_post, var_from_invgamma*V_post)

    return [[a_post, b_post], V_post, m_post, beta_draw]

def draw_posterior(a, b, m, V):
    var_from_invgamma = invgamma.rvs(a, 0, b, size=1)
    beta_draw = np.random.multivariate_normal(m, var_from_invgamma * V)
    return beta_draw
