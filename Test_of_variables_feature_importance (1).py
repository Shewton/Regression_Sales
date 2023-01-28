#!/usr/bin/env python
# coding: utf-8

# **TESTANDO VARIÁVEIS CONVERTIDAS EM CATEGÓRICAS(short tail)**

# In[ ]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as snsplt


# In[ ]:


import seaborn as sns


# In[ ]:


data = pd.read_csv("variáveis_categóricas_feature_importance_shortail - Sheet1.csv")
data.head()
data.shape


# In[ ]:



X = data.drop('ado_90d',axis=1)
X


# In[ ]:


y = data['ado_90d']
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)# Treinando modelo


# In[ ]:


model = RandomForestClassifier()


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


model.feature_importances_


# In[ ]:


importances = pd.Series(data=model.feature_importances_, index=[''])


# In[ ]:


sns.barplot(x=importances, y=importances.index, orient='h').set_title('Importância de cada feature')


# In[ ]:


model.score(X_train, y_train)


# **TESTANDO CORRELAÇÃO VARIÁVEIS DESCRITIVAS (short tail)**

# In[4]:


teste_90d = pd.read_csv(" ")
teste_60d = pd.read_csv("")
teste_30d = pd.read_csv("")
teste_15d = pd.read_csv("")


# In[ ]:


snsplt.heatmap(teste_90d.corr(),annot = True,fmt = '.2f',cmap='Blues')


# In[ ]:


snsplt.heatmap(teste_60d.corr(),annot = True,fmt = '.2f',cmap='Blues')


# In[ ]:


snsplt.heatmap(teste_30d.corr(),annot = True,fmt = '.2f',cmap='Blues')


# In[ ]:


snsplt.heatmap(teste_15d.corr(),annot = True,fmt = '.2f',cmap='Blues')


# **TESTANDO CORRELAÇÃO VARIÁVEIS DESCRITIVAS (short tail)(V2) (COM DADOS DE BUNDLE, SKU`S WITH SALES)**

# In[ ]:


teste_90d = pd.read_csv("Base_teste02.csv")
X = teste_90d.drop('stock_total',axis=1)


# In[ ]:


snsplt.heatmap(X.corr(),annot = True,fmt = '.2f',cmap='Blues')


# **TESTANDO RFE VARIÁVEIS DESCRITIVAS (V2) (COM DADOS DE BUNDLE, SKU`S WITH SALES) (dropando variáveis) (short tail)**

# In[ ]:


import numpy as np


# In[ ]:


teste_2 = pd.read_csv("Base_teste02 - Base_ajustadav3.csv")
teste_2.info()


# In[ ]:


teste_2[''] = teste_2[''].values.astype(np.int64)
teste_2[''] = teste_2[''].values.astype(np.int64)
teste_2[''] = teste_2[''].values.astype(np.int64)
teste_2[''] = teste_2[''].values.astype(np.int64)
teste_2[''] = teste_2[''].values.astype(np.int64)
teste_2[''] = teste_2[''].values.astype(np.int64)
teste_2[''] = teste_2[''].values.astype(np.int64)
teste_2[''] = teste_2[''].values.astype(np.int64)


# In[ ]:


teste_2.info()


# In[ ]:


X = teste_2.drop([''],axis=1)


# In[ ]:


y = teste_2['']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[ ]:


X_train


# In[ ]:


get_ipython().system('pip install sklearn')


# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=5000)


# In[ ]:


from sklearn.feature_selection import RFE


# In[ ]:


rfe = RFE(model,n_features_to_select=5)


# In[ ]:


fit = rfe.fit(X_train, y_train)


# In[ ]:


print("more_safe_variables:{}".format(fit.n_features_))


# In[ ]:


cols = fit.get_support(indices=True)
teste_2.iloc[:,cols]


# **DECISION TREE (MIDTAIL -> SHORTTAIL)**

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('pylab', 'inline')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error


# In[ ]:


base = pd.read_csv("midtail_shorttail_decisiontree_v0.csv")


# In[ ]:


base.head()


# In[ ]:


base_ajustada = base.drop([''],axis=1)


# In[ ]:


base_ajustada.head()


# In[ ]:


base_ajustada.info()


# In[ ]:


base_ajustada[''] = base_ajustada[''].values.astype(np.int64)
base_ajustada[''] = base_ajustada[''].values.astype(np.int64)
base_ajustada[''] = base_ajustada[''].values.astype(np.int64)
base_ajustada[''] = base_ajustada[''].values.astype(np.int64)
base_ajustada[''] = base_ajustada[''].values.astype(np.int64)
base_ajustada[''] = base_ajustada[''].values.astype(np.int64)


# In[ ]:


base_ajustada.info()


# In[ ]:


X = base_ajustada.drop('',axis=1)
X


# In[ ]:


y = base_ajustada['']
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5, random_state=42) # Treinando modelo


# In[ ]:


tree = DecisionTreeRegressor(max_depth=4, random_state=42)
##min_samples_leaf=15
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5,random_state=42) # Treinando modelo
tree.fit(X_train, y_train)

p = tree.predict(X_test)

np.sqrt(mean_squared_error(y_test, p))


# In[ ]:


pylab.figure(figsize=(100,100))
plot_tree(tree, feature_names = X_train.columns,fontsize=30)


# In[ ]:


##Observações
## conclusão 1


# **DECISION TREE V2 (sem gross_orders, items_with_sales) -> colocando dividend**

# In[ ]:


base_ajustada2 = base.drop([''],axis=1)


# In[ ]:


##base_ajustada['gross_orders_90d'] = base_ajustada['gross_orders_90d'].values.astype(np.int64)
base_ajustada2[''] = base_ajustada2[''].values.astype(np.int64)
base_ajustada2[''] = base_ajustada2[''].values.astype(np.int64)
base_ajustada2[''] = base_ajustada2[''].values.astype(np.int64)
base_ajustada2[''] = base_ajustada2[''].values.astype(np.int64)
base_ajustada2[''] = base_ajustada2[''].values.astype(np.int64)


# In[ ]:


base_ajustada2.info()


# In[ ]:


base_ajustada2.head()


# In[ ]:


X = base_ajustada2.drop('',axis=1)
X


# In[ ]:


y = base_ajustada['']
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5, random_state=42)


# In[ ]:


tree = DecisionTreeRegressor(max_depth=4, random_state=42)
##min_samples_leaf=15
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5,random_state=42) # Treinando modelo
tree.fit(X_train, y_train)

p = tree.predict(X_test)

np.sqrt(mean_squared_error(y_test, p))


# In[ ]:


pylab.figure(figsize=(100,100))
plot_tree(tree, feature_names = X_train.columns,fontsize=30)
plt.savefig('teste', dpi=100)


# In[ ]:


##adgmv + orders_ads tem um ado entre 312 e 212 = Variáveis de impacto


# **Decision Tree_V3 - Variáveis categóricas**

# In[ ]:


get_ipython().system('pip install graphviz')


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydot
import graphviz


# In[ ]:


import pydot_ng


# In[ ]:


base = pd.read_csv(".csv")
base


# In[ ]:


X = base.drop('', axis=1)
X


# In[ ]:


y = base['']
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5, random_state=42)


# In[ ]:


modelo_v1 = DecisionTreeClassifier(max_depth = 4, max_features = None, criterion = 'entropy', min_samples_leaf = 1, min_samples_split = 2)


# In[ ]:


modelo_v1.fit(X_train, y_train)


# In[ ]:


arquivo = '/content/sample_data/tree_modelo_v1.png'


# In[ ]:


export_graphviz(modelo_v1, out_file = arquivo, feature_names = X_train.columns)
with open(arquivo) as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# In[ ]:





# In[ ]:




