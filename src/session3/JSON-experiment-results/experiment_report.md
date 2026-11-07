```
# Speculative RAG experiments - 4 questions, stage=structure, retrieval k=10, clusters k=3

==============================================================================
STAGE: CLUSTER STRUCTURE (no LLM)  - only questions whose answer was retrieved
==============================================================================

[1] Intrinsic quality at k=3 (variable-k algorithms choose their own k)
    silhouette/stability: higher is better; davies_bouldin: lower is better; balance: 1 = even sizes
algorithm                  #clust  silhouette  dav-bouldin  stability  balance  singletons  ms   best-clust-prec
-------------------------  ------  ----------  -----------  ---------  -------  ----------  ---  ---------------
kmeans_current             3.0     0.146       1.519        0.725      0.88     16.7%       3.4  0.33           
kmeans_l2                  3.0     0.146       1.519        0.725      0.88     16.7%       4.9  0.33           
agglo_ward                 3.0     0.146       1.664        0.784      0.96     0.0%        0.3  0.38           
agglo_cosine               3.0     0.234       0.649        0.861      0.58     66.7%       0.3  0.31           
spectral                   3.0     0.151       1.623        0.788      0.95     0.0%        3.2  0.42           
gmm                        3.0     0.209       0.893        0.529      0.66     50.0%       5.5  0.34           
dbscan                     1.0     -           -            1.000      0.00     0.0%        0.9  0.25           
kcenter                    3.0     0.232       0.653        0.885      0.58     66.7%       0.1  0.31           
rank_roundrobin (control)  3.0     -0.052      2.374        0.437      0.99     0.0%        0.0  0.50           
random (control)           3.0     -0.072      2.357        0.000      0.99     0.0%        0.0  0.50           

[2] What each algorithm does to the evidence at k=3, per input mode
    coverage ~ must stay high. diversity ~ higher = more different drafts. unique ~ 1.0 = no duplicate drafts.
algorithm        mode                n  coverage  draft-support  diversity  unique  #drafts  chars/draft
---------------  ------------------  -  --------  -------------  ---------  ------  -------  -----------
kmeans_current   subsets             2  100.0%    100.0%         0.472      1.00    3.0      4297       
kmeans_current   clusters_only       2  100.0%    66.7%          1.000      1.00    3.0      2764       
kmeans_current   clusters_plus_top2  2  100.0%    100.0%         0.704      1.00    3.0      4037       
kmeans_l2        subsets             2  100.0%    100.0%         0.472      1.00    3.0      4297       
kmeans_l2        clusters_only       2  100.0%    66.7%          1.000      1.00    3.0      2764       
kmeans_l2        clusters_plus_top2  2  100.0%    100.0%         0.704      1.00    3.0      4037       
agglo_ward       subsets             2  100.0%    100.0%         0.651      1.00    3.0      4457       
agglo_ward       clusters_only       2  100.0%    66.7%          1.000      1.00    3.0      2764       
agglo_ward       clusters_plus_top2  2  100.0%    100.0%         0.724      1.00    3.0      4037       
agglo_cosine     subsets             2  100.0%    100.0%         0.333      1.00    3.0      4252       
agglo_cosine     clusters_only       2  100.0%    33.3%          1.000      1.00    3.0      2764       
agglo_cosine     clusters_plus_top2  2  100.0%    100.0%         0.685      1.00    3.0      4037       
spectral         subsets             2  100.0%    100.0%         0.611      1.00    3.0      4561       
spectral         clusters_only       2  100.0%    66.7%          1.000      1.00    3.0      2764       
spectral         clusters_plus_top2  2  100.0%    100.0%         0.722      1.00    3.0      4037       
gmm              subsets             2  100.0%    100.0%         0.413      1.00    3.0      4348       
gmm              clusters_only       2  100.0%    33.3%          1.000      1.00    3.0      2764       
gmm              clusters_plus_top2  2  100.0%    100.0%         0.697      1.00    3.0      4037       
dbscan           subsets             2  100.0%    100.0%         0.500      1.00    3.0      2729       
dbscan           clusters_only       2  100.0%    100.0%         0.000      1.00    1.0      8291       
dbscan           clusters_plus_top2  2  100.0%    100.0%         0.000      1.00    1.0      8291       
kcenter          subsets             2  100.0%    100.0%         0.333      1.00    3.0      4236       
kcenter          clusters_only       2  100.0%    33.3%          1.000      1.00    3.0      2764       
kcenter          clusters_plus_top2  2  100.0%    100.0%         0.685      1.00    3.0      4037       
rank_roundrobin  subsets             2  100.0%    100.0%         0.690      1.00    3.0      4350       
rank_roundrobin  clusters_only       2  100.0%    66.7%          1.000      1.00    3.0      2764       
rank_roundrobin  clusters_plus_top2  2  100.0%    100.0%         0.726      1.00    3.0      4037       
random           subsets             2  100.0%    100.0%         0.690      1.00    3.0      4258       
random           clusters_only       2  100.0%    66.7%          1.000      1.00    3.0      2764       
random           clusters_plus_top2  2  100.0%    100.0%         0.726      1.00    3.0      4037       

[3] Choosing k (cluster count) - silhouette | coverage(subsets) | coverage(clusters_only) | stability
algorithm        metric               k=2     k=3     k=4     k=5   
---------------  -------------------  ------  ------  ------  ------
kmeans_current   silhouette           0.287   0.146   0.122   0.112 
                 coverage subsets     100.0%  100.0%  100.0%  100.0%
                 coverage clust-only  100.0%  100.0%  100.0%  100.0%
                 stability            0.772   0.725   0.751   0.688 
kmeans_l2        silhouette           0.287   0.146   0.122   0.112 
                 coverage subsets     100.0%  100.0%  100.0%  100.0%
                 coverage clust-only  100.0%  100.0%  100.0%  100.0%
                 stability            0.772   0.725   0.575   0.672 
agglo_ward       silhouette           0.287   0.146   0.150   0.125 
                 coverage subsets     100.0%  100.0%  100.0%  100.0%
                 coverage clust-only  100.0%  100.0%  100.0%  100.0%
                 stability            0.715   0.784   0.831   0.804 
agglo_cosine     silhouette           0.294   0.234   0.144   0.116 
                 coverage subsets     100.0%  100.0%  100.0%  100.0%
                 coverage clust-only  100.0%  100.0%  100.0%  100.0%
                 stability            0.950   0.861   0.802   0.771 
spectral         silhouette           0.249   0.151   0.127   0.159 
                 coverage subsets     100.0%  100.0%  100.0%  100.0%
                 coverage clust-only  100.0%  100.0%  100.0%  100.0%
                 stability            0.795   0.788   0.702   0.713 
gmm              silhouette           0.294   0.209   0.115   0.122 
                 coverage subsets     100.0%  100.0%  100.0%  100.0%
                 coverage clust-only  100.0%  100.0%  100.0%  100.0%
                 stability            0.439   0.529   0.450   0.426 
kcenter          silhouette           0.294   0.232   0.118   0.097 
                 coverage subsets     100.0%  100.0%  100.0%  100.0%
                 coverage clust-only  100.0%  100.0%  100.0%  100.0%
                 stability            0.898   0.885   0.769   0.802 
rank_roundrobin  silhouette           -0.054  -0.052  -0.141  -0.061
                 coverage subsets     100.0%  100.0%  100.0%  100.0%
                 coverage clust-only  100.0%  100.0%  100.0%  100.0%
                 stability            0.325   0.437   0.420   0.438 
random           silhouette           0.004   -0.072  -0.109  -0.141
                 coverage subsets     100.0%  100.0%  100.0%  100.0%
                 coverage clust-only  100.0%  100.0%  100.0%  100.0%
                 stability            -0.102  0.000   0.025   -0.125

How to read it: an algorithm is only worth the switch if it keeps COVERAGE at least as high as
kmeans_current while raising DIVERSITY or STABILITY, and beats both controls (rank_roundrobin, random).
```
