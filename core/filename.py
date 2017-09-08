""" Provides a function for file naming
"""

def fname(K, hbar, theta, max_order, L):
    """
    """
    path = '../data/' + 'K' + str(K).replace(".","") + "/"
    name1 = 'hbar_' + str(hbar) + "_theta_" + str(theta)
    name2 = "_max_order_" + str(max_order) + "_L_" + str(L)
    return path + name1 + name2