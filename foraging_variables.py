#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:43:59 2019

@author: sergej
"""


#import os
import numpy as np
from copy import deepcopy
import pandas as pd

# output directory
#os.mkdir('')
#out_dir = "/home/sergej/Desktop/"

## designs variables
f_n = range(0,24)                   # forest number
d_n = range(0,5)                    # day number (trials)

## forest looping
for i_forest in f_n:
    ## conditions: 0 = no-threat, 1 = threat
    c = int(np.random.choice([0,1],1))
    ## q values for threat condition
    if c == 1:
        q_t = float(np.random.choice([0.1,0.2,0.3],1))          # threat prob
        # foraging q for threat condition
        if q_t == 0.3:
            q_fg = 0.1                                          # good env
            q_fb = float(np.random.choice([0.5,0.6],1))         # bad env
        elif q_t == 0.2:
            q_fg = float(np.random.choice([0.1,0.2],1))         # good env
            q_fb = float(np.random.choice([0.5,0.6,0.7],1))     # bad env
        elif q_t == 0.1:
            q_fg = float(np.random.choice([0.1,0.2,0.3],1))     # good env
            q_fb = float(np.random.choice([0.5,0.6,0.7,0.8],1)) # bad env
    ## q values for no-threat condition
    else:
        q_t = float(0)
        q_fg = float(np.random.choice([0.1,0.2,0.3,0.4],1))     # good env
        q_fb = float(np.random.choice([0.5,0.6,0.7,0.8,0.9],1)) # bad env
    ## p values for environments (used in MDP): [good, bad]
    p_f = np.array((round((1-q_fg-q_t),1), round((1-q_fb-q_t),1)))
    
    # other variables
    e_init = int(np.random.choice(np.arange(2,5),1))    # initial lp
    f_gain = int(np.random.choice(np.arange(0,5),1))    # foraging gain
    
    ## MDP variables
    ## variable frames
    c_w = np.array((-1, -1))            # cost waiting
    p_e = np.array((0.5,0.5))           # probability of being in the environments
    g_f = np.array((f_gain,f_gain))     # gain when foraging success
    c_f = np.array((-2,-2))             # cost when foraging unsuccessful
    p_f = np.array((round((1-q_fg-q_t),1),round((1-q_fb-q_t),1)))   # p values for environments (good,bad)
    q_f = np.array((q_fg,q_fb))         # q value foraging unsuccessful (good, bad)
    c_t = np.array((-3,-3))             # cost when encountering threat
    qta = np.array((q_t,q_t))           # probability of encountering threat
    pol_mat         = np.empty((5,6*2))
    pol_mat[:]      = np.nan
    rew_mat         = np.matrix((np.zeros((6,6*2))))
    rew_mat[:,0]    = -1
    rew_mat[:,6]    = -1
    rew_wai         = deepcopy(rew_mat)
    rew_for         = deepcopy(rew_mat)
    rew_cur         = np.zeros(2)
    trans_vec       = np.zeros((1,(6*2)))
    
    ## day looping & variable monitoring
    for i_day in d_n:
        ## variables: *targetted predictor variables*
        w = int(np.random.choice([1,2],1))  # *weather type*: 1 = good, 2 = bad
        if w == 1:
            p = p_f[0]                      # *p value*
            qT= q_t                         # q threat encounter
            qF= q_fg                        # q foraging non-success
            g = f_gain                      # *foraging gain*
            ev= p*g+(qF*-2)+(qT*-3)         # *expected value*
            #dp= i_day                         # *days past*
        else:
            p = p_f[1]                      # *p value*
            qT= q_t                         # q threat encounter
            qF= q_fb                        # q foraging non-success
            g = f_gain                      # *foraging gain*
            ev= p*g+(qF*-2)+(qT*-3)         # *expected value*
            #dp= i_day                         # *days past*
        
        ## optimal policy for 5 transitions (time horizon) and 2 environments (options)
        for i_sta in range(1,6):
            
            for i_env in range(0,2):
                
                ## transitioning
                # wait action - cost only
                trans_wait = deepcopy(trans_vec)
                trans_wait[0,i_sta + c_w[i_env]]      = 1*p_e[0]  # good env
                trans_wait[0,i_sta + c_w[i_env]+6]    = 1*p_e[1]  # bad env
                # foraging action - gain
                trans_fora = deepcopy(trans_vec)
                if i_sta + g_f[i_env]     >= 6:
                    g_ind = 6-1
                else:
                    g_ind = i_sta + g_f[i_env]
                trans_fora[0,g_ind]       = p_f[i_env]*p_e[0]     # good env
                trans_fora[0,g_ind+6]     = p_f[i_env]*p_e[1]     # bad env
                # foraging action - cost
                if i_sta + c_f[i_env]      <= 0:
                    c_ind = 0
                else:
                    c_ind = i_sta+c_f[i_env]
                trans_fora[0,c_ind]       = (q_f[i_env])*p_e[0]   # good env
                trans_fora[0,c_ind+6]     = (q_f[i_env])*p_e[1]   # bad env
                # foraging action - predation
                if i_sta + c_t[i_env]      <= 0:
                    t_ind = 0
                else:
                    t_ind = i_sta+c_t[i_env]
                trans_fora[0,t_ind]       = (qta[i_env])*p_e[0]   # good env
                trans_fora[0,t_ind+6]     = (qta[i_env])*p_e[1]   # bad env
                ## reward obtained
                rew_cur[0] = trans_wait * rew_mat[i_day,:].T
                rew_cur[1] = trans_fora * rew_mat[i_day,:].T
                rew_max1 = np.max(rew_cur)
                rew_max2 = np.argmax(rew_cur)   # 0: wait; 1: forage
                if rew_cur[0] == rew_cur[1]:
                    rew_max2 = 2                # 2: indifference
                pol_mat[i_day, i_sta+i_env*6]     = rew_max2
                rew_mat[i_day+1, i_sta+i_env*6]   = rew_max1
                rew_wai[i_day+1, i_sta+i_env*6]   = rew_cur[0]
                rew_for[i_day+1, i_sta+i_env*6]   = rew_cur[1]
    # MDP reward matrices
    r_wai = pd.DataFrame(rew_wai)               # reward waiting
    r_for = pd.DataFrame(rew_for)               # reward foraging
    r_con = pd.concat([r_for,r_wai], axis=1)    # concatenated rewards (foraging, waiting)
    
    r_mat = r_con.iloc[:,:6]-r_con.iloc[:,12:18]     # *optimal policies* (states*continuous energy state)
    