def plot_cluster(subset=['travel', 'tech', 'fashion', 'fitness', 'food',
                        'age', 'studies', 'gender', 'ig_since', 'ig_time_daily'],
                 clusters = 7):
    # coding: utf-8

    # In[123]:


    import imp
    import data_preparation
    imp.reload(data_preparation)
    df, df_sdt, ents_srt, exts_srt, means, stds, ents, entropy = data_preparation.get()

    print()
    print("****")
    print("Cluster analysis using the following subset: ", subset)
    print("Number of clusters = ", clusters)
    print("****")
    print()

    # In[228]:


    # Subset for clustering

    # len("num-visit_sponsored_page;num-interact_w_sponsored_brand;num-do_referral_brand;num-cross_information;num-direct_brand_communication;num-did_buy_via_infl;num-did_referred_sponsored_item;num-wannabe_infl".split(";"))

    #df_sdt = df_sdt[df_sdt.columns.values[-20:]]
    #df_sdt = 
    df_subset = df_sdt[subset].copy()
    df_subset.head()


    # In[229]:


    X = df_subset.values
    print("Matrix size: ", X.shape)


    # In[230]:


    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    # PCA ANALYSIS
    pca = PCA(n_components=X.shape[1]).fit(X)
    ninty_cutoff, seventyfive_cutoff = False, False
    for ix, variance in enumerate(pca.explained_variance_ratio_.cumsum()):
        if ix == 3:
            print("Variance explained in 3 dimensions: ", variance)
        if (variance >= 0.75) & (not seventyfive_cutoff):
            seventyfive_cutoff = True
            print('0.75 cutoff -> ', ix)
        if (variance >= 0.9) & (not ninty_cutoff):
            ninty_cutoff = True
            print('0.90 cutoff -> ', ix)


    # In[231]:


    from mpl_toolkits.mplot3d import axes3d, Axes3D

    pca_3d = PCA(n_components=3).fit_transform(X)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2], s=300, c='r', marker='o')
    fig.show();


    # In[232]:


    pca_2d = PCA(n_components=2).fit_transform(X)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(pca_2d[:, 0], pca_2d[:, 1], s=300, c='r', marker='o')
    fig.show();


    # In[120]:


    distortions = []
    for i in range(1, int(30)):
        # print("Fitting {} clusters".format(i))
        distortions.append(KMeans(n_clusters=i, init='k-means++').fit(PCA(n_components=3).fit_transform(X)).inertia_)
    print("Average inertia: {}".format(sum(distortions)/len(distortions)))
    print()

    plt.figure(figsize=(8,8));
    plt.plot(range(1, int(30)), distortions, marker='o');
    plt.show();


    # In[233]:


    import pandas as pd
    import numpy as np

    X_to_be_used = X

    clustering = pd.DataFrame(data=KMeans(n_clusters=clusters, init='k-means++').fit_predict(X_to_be_used))

    reduced_data_plot = PCA(n_components=2).fit_transform(X)
    kmeans = KMeans(n_clusters=clusters, init='k-means++').fit(reduced_data_plot)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data_plot[:, 0].min() - 1, reduced_data_plot[:, 0].max() + 1
    y_min, y_max = reduced_data_plot[:, 1].min() - 1, reduced_data_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(8,8))
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect='auto', origin='lower')

    plt.plot(reduced_data_plot[:, 0], reduced_data_plot[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show();


    # In[122]:


    plt.figure(figsize=(8,8))
    dendrogram(
                linkage(X_to_be_used),
                orientation='right',
                distance_sort='descending',
                show_leaf_counts=True,
            )
    plt.show();
    plt.tight_layout()


    # In[261]:


    pca_3d_results = PCA(n_components=3).fit_transform(X)
    pca_2d_results = PCA(n_components=2).fit_transform(X)
    df_subset['pca_3d_1'] = pca_3d_results[:, 0]
    df_subset['pca_3d_2'] = pca_3d_results[:, 1]
    df_subset['pca_3d_3'] = pca_3d_results[:, 2]
    df_subset['pca_2d_1'] = pca_2d_results[:, 0]
    df_subset['pca_2d_2'] = pca_2d_results[:, 1]
    df_subset['k_full'] = KMeans(n_clusters=clusters, init='k-means++').fit_predict(X)
    df_subset['k_pca'] = KMeans(n_clusters=clusters, init='k-means++').fit_predict(PCA(n_components=0.90).fit_transform(X))
    # df_subset.head()


    # In[263]:


    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(df_subset['pca_2d_1'].values, 
            df_subset['pca_2d_2'].values, 
            s=300, 
            c=df_subset['k_full'].values, 
            marker='o')
    fig.show();


    # In[266]:


    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_subset['pca_3d_1'].values, 
            df_subset['pca_3d_2'].values, 
            df_subset['pca_3d_3'].values,
            c=df_subset['k_full'].values,
            s=300, marker='o')
    fig.show();


    # In[235]:

    import seaborn as sns

    corr = df_subset[[c for c in df_subset.columns.values if 'pca' not in c]].corr() 
    plt.figure(figsize=(8, 8));
    sns.set(font_scale=1.4)
    sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
                cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
                annot=True, annot_kws={"size": 14}, square=True);
    plt.show();


    # In[258]:


    from matplotlib import pyplot as plt

    def plot_frequencies(k, df_to_plot, df_reference):
        rows = int(df_to_plot.shape[1]/3)+(df_to_plot.shape[1]%3>0)
        cols = 3
        fig, ax = plt.subplots(rows, cols, figsize=(15,20))
        fig.suptitle("k = {}".format(k), fontsize=24)
        answer = 0
        for row in range(rows):
            for col in range(cols):
                if (answer == df_to_plot.shape[1]):
                    break
                feature_name = df_to_plot.columns.values[answer]
                cc = df_to_plot[feature_name]
                cc_ref = df_reference[feature_name]
                if feature_name == "k":
                    continue
                # For distribution -> sns.distplot(cc, ax=ax[row, col])
                # sns.countplot(cc, ax=ax[row, col])
                graph_df = cc_ref.value_counts(normalize=True).apply(lambda x: round(x,2)).rename('all').to_frame()
                graph_df = graph_df.join(cc.value_counts(normalize=True).apply(lambda x: round(x,2)).rename("k={}".format(k)).to_frame())
                graph_df = graph_df.fillna(0)
                graph_df.index = [round(x, 2) for x in graph_df.index.values]
                graph_df.plot(kind='bar', ax=ax[row, col], rot=0, 
                            title=feature_name, legend=answer==0,
                            fontsize=10)
                if answer == 0:
                    ax[row, col].legend(loc='best', prop={'size': 12})
                box_content = "{} {} ± {} \n {} {} ± {}".format(
                    "" if answer != 0 else "k={}: ".format(k),
                    round(cc.mean(), 2),
                    round(cc.std(), 2),
                    "" if answer != 0 else "all: ".format(k),
                    round(cc_ref.mean(), 2),
                    round(cc_ref.std(), 2),
                )
                ax[row, col].text(0.5, 0.5, box_content, 
                                fontsize=13, 
                                horizontalalignment='center', 
                                verticalalignment='center', 
                                transform=ax[row, col].transAxes,
                                bbox=dict(facecolor='red', alpha=0.5))
                answer += 1
        fig.tight_layout()
        plt.subplots_adjust(top=0.85)
    plt.show();

    # In[260]:


    for k in range(clusters):
        df_reference = (df_subset[[c for c in df_subset.columns.values if 'pca' not in c]])
        df_to_be_plot = df_reference[df_subset.k_full == k]
        plot_frequencies(k, df_to_be_plot, df_reference)

