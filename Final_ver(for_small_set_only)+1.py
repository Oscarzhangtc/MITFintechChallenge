#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)


# **read_data_small** is the function to read in the small dataset about 30 MB

# In[2]:


def read_data_small():
    X_train = pd.read_csv("data_small/X_train_small.csv")
    X_test = pd.read_csv("data_small/X_test_small.csv")
    y_train = np.asarray(pd.read_csv("data_small/y_train_small.csv", header=None)[0])
    return X_train, X_test, y_train


# **read_data_big** is the function to read in the big dataset about 100 MB

# In[3]:


def read_data_big():
    X_train = pd.read_csv("data_big/X_train_big.csv")
    X_test = pd.read_csv("data_big/X_test_big.csv")
    y_train = np.asarray(pd.read_csv("data_big/y_train_big.csv", header=None)[0])
    return X_train, X_test, y_train


# **read_data** is the function to read in the whole dataset about 1.5 G

# In[4]:


def read_data():
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = np.asarray(pd.read_csv("data/y_train.csv", header=None)[0])
    return X_train, X_test, y_train


# # Insert Your Code Here

# In[5]:


get_ipython().run_cell_magic('time', '', 'X_train, X_test, y_train = read_data_small()\ndata = X_train')


# In[6]:


#type1and2 = data[(data["class"] == 1) ^ (data["class"] == 2)]
type1and2 = data


# In[19]:


sort_by = ["member", "user", "endUserRef","obId"]
type1and2_person = np.unique(type1and2[sort_by].values.astype(str), axis=0)


# In[20]:


all_person = data[sort_by].values.astype(str)
type1and2_person_activities = data[[x in type1and2_person for x in all_person]]


# In[21]:


len(data)


# In[22]:


len(type1and2_person)


# In[36]:


y_val = pd.DataFrame()
y_val["index"] = data.index
y_val["true_class"] = y_train
y_val.set_index("index")

y_pred = pd.DataFrame()
y_pred["index"] = data.index
y_pred["predicted_class"] = 0
y_pred.set_index("index")
print("")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'a = type1and2_person_activities.groupby(["member", "user", "endUserRef","obId"])\nshort_time = 1000\n\nfor k in range(len(type1and2_person)):\n    if k % 10 == 0:\n        print("k = ",k)\n    \n    person_0 = a.get_group(tuple(type1and2_person[k]))[["timestamp","type","volume", "operation", "isBid", "orderId",\'isBuyer\',\'bidOrderId\', \'askOrderId\', "isAggressor"]]\n    if len(person_0) > len(data) /500:\n        continue\n    \n    person_0_orderbook = {}\n    volume_ls = []\n#    class_ls = []\n    for x in person_0.iterrows():\n        action = x[1]\n        if action.type == "ORDER":\n            pos = int(action.volume)\n            if action.isBid == False:\n                pos = pos * -1\n            if action.operation == "CANCEL":\n                del person_0_orderbook[action.orderId]\n#                print("CANCEL", pos, end=",  ")\n            else:\n                person_0_orderbook[action.orderId] = pos\n#            if action.operation == "INSERT":\n#                print("INSERT", pos, end=",  ")            \n            volume = list(person_0_orderbook.values())\n            volume_ls.append(volume)\n#            class_ls.append(action[\'class\'])\n#            print(x[0], volume, "true class", action[\'class\'], "predict class", -(x[0] in v))\n        else:\n            pos = int(action.volume)\n            if action.isBuyer == False:\n                pos = pos * -1\n#            print("TRADE", pos, x[0], "true class", action[\'class\'], "predict class", -(x[0] in v))\n\n    volume_history = pd.DataFrame()\n    volume_history[\'time\'] = person_0[person_0["type"]=="ORDER"].index\n    volume_history[\'volume\'] = volume_ls\n    volume_history[\'long_pos\'] = [[y for y in x if y>0] for x in volume_ls]\n    volume_history[\'short_pos\'] = [[y for y in x if y<0] for x in volume_ls]\n    volume_history[\'long_pos_total\'] = [sum([y for y in x if y>0]) for x in volume_ls]\n    volume_history[\'short_pos_toal\'] = [sum([y for y in x if y<0]) for x in volume_ls]\n    volume_history = volume_history.set_index(\'time\')\n    volume_threshold = np.mean([abs(y) for x in volume_ls for y in x]) * 0.8\n    #volume_history\n    #plt.hist([abs(y) for x in volume_ls for y in x])\n\n    canceled_order = person_0[(person_0["operation"] == "CANCEL") & (person_0["volume"] != 0)]\n    person_0_trades = person_0[person_0["type"]=="TRADE"]\n    person_0_insert = person_0[(person_0["operation"]=="INSERT") & (person_0["orderId"].isin(canceled_order[\'orderId\'].values))]\n\n    canceled_order_time = pd.DataFrame()\n    canceled_order_time["endtime"] = canceled_order.index\n    canceled_order_time["orderId"] = canceled_order.orderId.values.astype(str)\n    canceled_order_time["isBid"] = canceled_order.isBid.values\n    person_0_insert_time = pd.DataFrame()\n    person_0_insert_time["starttime"] = person_0_insert.index\n    person_0_insert_time["orderId"] = person_0_insert.orderId.values.astype(str)\n\n    canceledordertimes = pd.concat([person_0_insert_time.set_index("orderId"),canceled_order_time.set_index("orderId")], axis=1, join=\'inner\')\n    canceled_order = pd.concat([canceledordertimes,canceled_order.set_index("orderId")], axis=1, join=\'inner\')\n\n    start = np.array(canceled_order[\'starttime\'].values)\n    end = np.array(canceled_order[\'endtime\'].values)\n    trade_ind = np.array(person_0_trades.index)\n\n    ls = set()\n    #Can be optimized\n    for i in range(len(start)):\n        for j in range(len(trade_ind)):\n    #        print(i, j, trade_ind[j] > start[i] ,trade_ind[j] < end[i] , end[i] - trade_ind[j] < 1000 )\n            if trade_ind[j] > start[i] and trade_ind[j] < end[i] and end[i] - trade_ind[j] < short_time and end[i]-start[i] < short_time * 5:\n    #            print("One")\n    #            print([trade_ind[j],start[i], end[i]])\n    #            print(abs(canceled_order.iloc[i].volume), volume_threshold)\n                if abs(canceled_order.iloc[i].volume) > 0.55*volume_threshold and person_0_trades.iloc[j][\'isBuyer\'] != canceledordertimes.iloc[i]["isBid"] and person_0_trades.iloc[j][\'isAggressor\'] == False:\n    #                print("Two")\n                    id = (person_0_trades.iloc[j].bidOrderId if person_0_trades.iloc[j][\'isBuyer\']==True else person_0_trades.iloc[j].askOrderId)\n                    sold = person_0[(person_0["operation"]=="INSERT") & (person_0["orderId"] == id)]\n    #                print(sold.index)\n                    other_trade_index = person_0_trades.iloc[np.where([(person_0_trades["bidOrderId"] == id) ^ (person_0_trades["askOrderId"] == id)])[1]].index\n                    ls.update([trade_ind[j],start[i], end[i]], sold.index,other_trade_index)\n    #                print("ls", ls)\n\n    v = np.array(list(ls))\n    v.sort()\n\n\n#    w = np.array(person_0[person_0["class"]!=0].index)\n\n#    print([x for x in v if x not in w], [x for x in w if x not in v], len(w), len(v))\n    \n\n   \n    action_ls = pd.DataFrame()\n    action_ls[\'index\'] = v\n    action_ls[\'action\'] = ""\n    action_ls[\'predicted_class\'] = 0\n    action_ls = action_ls.set_index("index")\n    action_ls\n    for x in person_0.loc[v].iterrows():\n        action = x[1]\n        if action.type == "ORDER":\n            if action.operation == "CANCEL":\n                action_ls[\'action\'].loc[x[0]] = "CANCEL"\n            if action.operation == "INSERT":\n                action_ls[\'action\'].loc[x[0]] = "INSERT"\n        else:\n            action_ls[\'action\'].loc[action.name] = "TRADE"\n\n    c_count = 0\n    i_count = 0\n    c_ls = []\n    for x in action_ls[::-1].iterrows():        \n        if x[1].action!="CANCEL":\n            if c_count!=0:\n                c_ls.append(x[0])\n                if x[1].action == "INSERT":\n                    i_count += 1\n                    if i_count == (c_count + 1):\n                        if c_count == 1:\n                            for y in c_ls:\n                                action_ls[\'predicted_class\'].loc[y]=1\n                        else:\n                            for y in c_ls:\n                                action_ls[\'predicted_class\'].loc[y]=2\n                        i_count = 0\n                        c_count = 0\n                        c_ls.clear()\n        else:\n            c_ls.append(x[0])\n            c_count += 1\n    action_ls = action_ls.replace(to_replace=0, method=\'ffill\')\n\n    if len(v) < 1000:\n        for x in action_ls.index:\n            y_pred.loc[x] = action_ls.loc[x]["predicted_class"]')


# In[55]:


from sklearn.metrics import cohen_kappa_score

def score(y_pred, y_true):
    y_pred_label = np.argmax(y_pred, axis=1)
    return cohen_kappa_score(y_pred_label, y_true)


# In[54]:


y_true = y_val["true_class"]

ls = []
t = []
for x in y_pred["predicted_class"].values:
    t = np.zeros(4,int)
    t[x] = 1
    ls.append(t)
y_pred_label = np.array(ls)
score(res, y_true)


# In[53]:





# In[42]:


np.count_nonzero(np.subtract(y_true,y_pred_label)), np.count_nonzero(y_true), len(y_true)


# In[52]:





# ### Submission Format
# 
# The classifier function wrote should return a 4d nparray with 4 columns. The columns are corresponding to the class labels: 0, 1, 2, 3. Please see examples below.

# In[ ]:


y_train_prob_pred


# In[ ]:


y_test_prob_pred


# ### Write test results to csv files

# Please rename your file to indicate which data set you are working with. 
# 
# - If you are using the small dataset: *y_train_prob_pred_small.csv* and *y_test_prob_pred_small.csv*
# - If you are using the small dataset: *y_train_prob_pred_big.csv* and *y_test_prob_pred_big.csv*
# - If you are using the original dataset: *y_train_prob_pred.csv* and *y_test_prob_pred.csv*

# In[ ]:


pd.DataFrame(y_train_prob_pred).to_csv("y_train_prob_pred.csv")
pd.DataFrame(y_test_prob_pred).to_csv("y_test_prob_pred.csv")

