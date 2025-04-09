# https://github.com/cdiego89phd/wtd-debiasing-RS-eval/blob/main/Debiasing%20Intervention.ipynb

import random
from recommenders.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL


def calculate_weights(df_training, df_mar, use_ideal, usrs, itms):
    # calculate P(u|O) for MAR
    p_u_O_MAR = calculate_p_u_o(df_mar, usrs, use_ideal)

    # calculate P(u|O) for MNAR
    p_u_O_MNAR = calculate_p_u_o(df_training, usrs, False)

    # calculate P(i|O) for MAR
    p_i_O_MAR = calculate_p_i_o(df_mar, itms, use_ideal)

    # calculate P(i|O) for MNAR
    p_i_O_MNAR = calculate_p_i_o(df_training, itms, False)

    w_users = initialize_weights(p_u_O_MNAR.keys())
    w_items = initialize_weights(p_i_O_MNAR.keys())

    w_users = exact_weight_calculation(p_u_O_MAR, p_u_O_MNAR, w_users)
    w_items = exact_weight_calculation(p_i_O_MAR, p_i_O_MNAR, w_items)

    # normalize weights
    w_users = norm(w_users)
    w_items = norm(w_items)

    return w_users, w_items


def initialize_weights(list_of):
    w = {}
    for e in list_of:
        w[e] = random.random()
    return w


def exact_weight_calculation(p_mar, p_mnar, gd_w):
    for ii in gd_w:
        gd_w[ii] = p_mar[ii] / p_mnar[ii]
    return gd_w


def norm(vect):
    sum_vect = sum(list(vect.values()))
    for key in vect:
        vect[key] = vect[key] / sum_vect
    return vect


def calculate_p_u_o(df, usrs, use_ideal):
    n = len(df)
    ideal_distr = 1 / len(usrs)
    p_u_o = {}
    for user in usrs:
        if use_ideal:
            p_u_o[user] = ideal_distr
        else:
            p_u_o[user] = len(df[df[DEFAULT_USER_COL] == user]) / n
        if p_u_o[user] == 0:
            p_u_o[user] = 0.0001
    return p_u_o


def calculate_p_i_o(df, itms, use_ideal):
    n = len(df)
    ideal_distr = 1 / len(itms)
    p_i_o = {}
    for item in itms:
        if use_ideal:
            p_i_o[item] = ideal_distr
        else:
            p_i_o[item] = len(df[df[DEFAULT_ITEM_COL] == item]) / n
        if p_i_o[item] == 0:
            p_i_o[item] = 0.0001
    return p_i_o
