def get():
    # coding: utf-8

    # In[1]:


    answers_file = 'answers_3.csv'
    questions_map = 'questions_map.txt'


    # In[30]:


    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")

    df = pd.read_csv(answers_file, delimiter=";")
    print("Dataframe size: ", df.shape)
    df = df.dropna()
    print("Purged dataframe size: ", df.shape)


    # In[31]:


    # Columns remapping

    mapping = open(questions_map, 'r')
    questions = mapping.read().split(";")
    mapping.close()
    mapping = {before: after for before, after in zip(df.columns.values, questions)}
    df = df.rename(columns=mapping)


    # In[32]:


    for c in df.columns.values:
        to_drop = []
        if df[c].value_counts().shape[0] == 1:
            print("Dropping {} because it's constant".format(c))
            to_drop.append(c)
        df = df.drop(columns=to_drop)


    # In[33]:


    # Re-map the categorical answers

    from sklearn import preprocessing
    les = {}

    for c in [c for c in df.columns.values if 'cat-' in c]:
        le = preprocessing.LabelEncoder()
        le.fit(df[c].values)
        df[c] = le.transform(df[c].values)
        les[c] = le


    # In[34]:


    # Features that are numerical, but needs to be handled
    for c in [c for c in df.columns.values if 'num-' in c]:
        cc = df[c].astype(str).str.extract('([0-9]{1})', expand=False).str.strip().astype(int)
        df[c] = cc


    # In[35]:


    # Custom mapping

    import re

    def map_followers(s):
        s = s.replace("1 Milione", "1000000").replace(".", "")
        p = re.compile("([0-9]+)")
        rng = [int(x) for x in p.findall(s)]
        return int(sum(rng)/len(rng))

    def map_age(s):
        if '14' in s:
            return 0
        if '19' in s:
            return 1
        if '30' in s:
            return 2
        print(s)
        return -1

    def map_ig_since(s):
        if 'Meno di 6' in s:
            return 0
        if 'Tra 6' in s:
            return 1
        if 'Tra 1' in s:
            return 2
        if 'Da più di 3' in s:
            return 3
        print(s)
        return -1

    def map_ig_time(s):
        if 'Meno di' in s:
            return 0
        if 'Tra 1' in s:
            return 1
        if 'Tra 2' in s:
            return 2
        if 'Più di' in s:
            return 3
        print(s)
        return -1

    def map_studies(s):
        s = s.lower()
        if ("triennale" in s) or ("laurea" in s) or ("biennale" in s):
            return 3
        if ("laurea specialistica" in s) or ("speciali" in s):
            return 4
        if ("master" in s) or ("dottorato" in s) or ("phd" in s):
            return 5
        if ("maturit" in s) or ("studente" in s) or ("liceo" in s):
            return 1
        if (s == "diploma") or ("superiori" in s) or ("diploma" in s):
            return 2
        if ("medie" in s) or ("licenza media" in s) or ("corso" in s) or ("qualifica" in s) or ("attestato" in s) or ("studiando" in s) or ("scuola" in s):
            return 0
        if ('make up artis' in s):
            return 0
        print(s)
        return -1

    def map_infl(s):
        if "N" in s:
            return 0
        if "S" in s:
            return 1
        print(s)
        return -1

    mapping_functions = {
        'demo-special-studies': map_studies,
        'demo-special-ig_time_daily': map_ig_time,
        'demo-special-ig_since': map_ig_since,
        'demo-special-age': map_age,
        'special-infl_followers': map_followers,
    }

    for key, function in mapping_functions.items():
        df[key] = df[key].apply(function).astype(int)


    # In[43]:


    # Focus on everything but multiple answers
    df_min = df[[c for c in df.columns.values if 'mul-' not in c]]

    # Let's define an entropy formula

    import numpy as np
    from math import log, e

    def entropy(labels, base=None):
        """ Computes entropy of label distribution. """
        n_labels = len(labels)
        if n_labels <= 1:
            return 0
        value, counts = np.unique(labels, return_counts=True)
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)
        if n_classes <= 1:
            return 0
        ent = 0.
        # Compute entropy
        base = e if base is None else base
        for i in probs:
            ent -= i * log(i, base)
        return ent

    means = {}
    stds = {}
    ents = {}
    exts = {}

    for c in df_min.columns.values:
        cc = df_min[c]
        c_nice = c.split("-")[-1]
        means[c_nice] = cc.mean()
        stds[c_nice] = cc.std()
        ents[c_nice] = entropy(cc.values)
        if 'num-' in c:
            exts[c_nice] = abs((cc.mean() - 3)/2)
            
    # Let's adapt features names for logging
    df_plot = df_min.rename(columns={c: c.split("-")[-1] for c in df.columns.values}).copy()


    # In[44]:


    df_mul = df[[c for c in df.columns.values if 'mul-' in c]]

    def map_s_discovered_infl(s):
        if 'interessi' in s:
            return 'interests'
        if 'amici' in s:
            return 'friends'
        if 'suggerimenti di Instagram' in s:
            return 'explore'
        if 'da altre pagine' in s:
            return 'algo_recommended'
        if 'dalla pagina di brand' in s:
            return 'sponsor'
        if 'sentito parlare o' in s:
            return 'wom'
        return '-1'

    def map_s_verticals(s):
        if 'Viaggi' in s:
            return 'travel'
        if 'Fashion' in s:
            return 'fashion'
        if 'Make' in s:
            return 'beauty'
        if 'Tech' in s:
            return 'tech'
        if 'Fitness' in s:
            return 'fitness'
        if 'Food' in s:
            return 'food'
        return '-1'

    def map_multiples(s, fun):
        qs = s.split(", ")
        rs = []
        for q in qs:
            rs.append(fun(q))
        return ','.join(rs)

    mapping_functions = {
        'mul-how_discovered_infl': map_s_discovered_infl,
        'mul-infl_verticals': map_s_verticals,
    }
    for key, function in mapping_functions.items():
        df_mul[key] = df_mul[key].apply(lambda x: map_multiples(x, function)).astype(str)

    set_keys = {}
    for c in df_mul.columns.values:
        set_keys[c] = [x for x in set(','.join(list(df_mul[c].values)).split(","))]
        
    for c, values in set_keys.items():
        for new_col in values:
            if new_col == "-1":
                continue
            rs = []
            for row in range(df_mul.shape[0]):
                rs.append((new_col in df_mul.iloc[row, df_mul.columns.get_loc(c)]) *1)
            df_mul[new_col] = rs

    df_mul = df_mul.drop(columns=[c for c in df_mul.columns.values if 'mul-' in c])

    df_sdt = df_plot.join(df_mul)


    # In[45]:


    df_sdt.head()


    # In[46]:


    df_sdt.describe()


    # In[54]:


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df_ = df_sdt.copy()
    df_[df_.columns] = scaler.fit_transform(df_[df_.columns])
    df_.head()

    ents_srt = sorted(ents.items(), key=lambda x: x[1])
    exts_srt = sorted(exts.items(), key=lambda x: x[1])[-5:]

    return df_sdt, df_, ents_srt, exts_srt, means, stds, ents, entropy