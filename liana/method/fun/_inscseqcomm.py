from liana._logging import _logg
import liana as li
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import numpy as np
import scipy.stats as stats
import decoupler as dc
from liana.method._pipe_utils import prep_check_adata
from liana.resource._select_resource import select_resource_tf_tg, select_resource_ppr_r_tf
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import time

from liana.testing._sample_anndata import generate_toy_adata

def compute_de(adata, groupby: str, method: str = "wilcoxon", **kwargs) -> dict:
    """
    Compute differentially expressed genes for each cluster of cells grouped by a chosen attribute.
    The method to identify differentially expressed genes is based on the wilcoxon test. Each gene is tested
    independently for each cluster of cells against the rest of the cells. The test is performed log fold change 
    expression values.

    Parameters
    ----------
    adata
        anndata.AnnData  (Annotated data object. Must have as raw level the log fold change expression matrix)
    groupby
        str (string that identifies which "observation" column of adata.obs to use for grouping cells and cluster them)
    method
        str, optional (Flag to chose which method is used to identify differential expressed genes. Defaults to "wilcoxon".)
        IF "wilcoxon" related **kwargs:
            alpha
                float, optional (significance threshold for the wilcoxon test. Defaults to 0.05)
            count_th
                int, optional (minimum gene expression threshold. Defaults to 1)
            ratio_th
                float, optional (minimum cell ratio in cluster expressing at least [count_th] gene threshold. Defaults to 0.25)

    Returns dedict (dictionary of dictionaries of pandas.Series for each cluster of cells. Each nested dictionary contains
                    DE (boolean pandas.Series of genes that are differentially expressed in the cluster), 
                    pval (float pandas.Series of pvalues for each gene in the cluster),
                    reliability (boolean pandas.Series of genes that are reliable in the cluster))
    """
    adata.obs['@label'] = adata.obs[groupby]
    labels = adata.obs['@label'].cat.categories
    genenames = adata.var_names
    if method == "wilcoxon":
        #assert "alpha" in kwargs, "[alpha] significance threshold parameter is required for wilcoxon method"
        #assert "count_th" in kwargs, "[count_th] minimum gene expression threshold parameter is required"
        #assert "ratio_th" in kwargs, "[ratio_th] minimum cell ratio in cluster expressing at least [count_th] gene threshold parameter is required"
        if "alpha" not in kwargs:
            print("[alpha] significance threshold parameter is not provided... using default value 0.05")
            alpha = 0.05
        else:
            alpha = kwargs["alpha"]
        if "count_th" not in kwargs:
            print("[count_th] minimum gene expression threshold parameter is not provided... using default value 1")
            count_th = 1
        else:
            count_th = kwargs["count_th"]
        if "ratio_th" not in kwargs:
            print("[ratio_th] minimum cell ratio in cluster expressing at least [count_th] gene threshold parameter is not provided... using default value 0.25")
            ratio_th = 0.25
        else:
            ratio_th = kwargs["ratio_th"]
        
        dedict = {label: {'DE':pd.Series([False]*len(genenames),index=genenames), \
                        'pval':pd.Series([1]*len(genenames),index=genenames), \
                        'reliability':pd.Series([False]*len(genenames),index=genenames)} for label in labels}
        # FOR EACH CLUSTER k \in K:
        for klabel in labels:
            _adata_k = adata[adata.obs['@label'].isin([klabel])]
            _adata_ref = adata[~adata.obs['@label'].isin([klabel])]
            
            # Filter the unreliable genes in both the cluster and the rest of the cells independently
            #TODO CHIEDERE A GIULIA PERCHE' FILTRARE PER CLUSTER E PER IL RESTO DELLE CELLULE IN MODO
            # INDIPENDENTE È GIUSTO? PERCHÈ NON FILTRARE TUTTE LE CELLE INSIEME?
            # cluster
            expressed_genes = _adata_k.raw.X.T > count_th
            present_genes = (np.sum(expressed_genes,axis=1)  / _adata_k.shape[0]) > ratio_th
            _reliable_gene_k = _adata_k.var_names[present_genes.A1]
            # rest of the cells
            expressed_genes = _adata_ref.raw.X.T > count_th
            present_genes = (np.sum(expressed_genes,axis=1)  / _adata_ref.shape[0]) > ratio_th
            _reliable_gene_ref = _adata_ref.var_names[present_genes.A1]
            # union of the two sets
            _reliable_genes = set(_reliable_gene_k).union(set(_reliable_gene_ref))
            # TODO UNRELIABLE GENES ARE NOT USED NOW
            __unreliable_genes = list(set(genenames) - _reliable_genes)
            print("Running cluster...: {}\nCluster cells shape: {} | all rest of cells shape: {}\n \
                reliable and unreliable number of genes: ({},{})".format(klabel, \
                                                                        _adata_k.raw.X.T.shape, \
                                                                        _adata_ref.raw.X.T.shape, \
                                                                        len(_reliable_genes), \
                                                                        len(__unreliable_genes)))
            # Compute wilcoxon test
            for gene in _reliable_genes:
                igene = adata.var_names.get_loc(gene)
                _, pvalue = stats.mannwhitneyu(_adata_k.raw.X.T[igene].A.flatten(), _adata_ref.raw.X.T[igene].A.flatten())
                # Bonferroni correction
                dedict[klabel]['DE'][gene] = pvalue < alpha/len(genenames)
                dedict[klabel]['pval'][gene] = pvalue
                dedict[klabel]['reliability'][gene] = True
        # END OF CLUSTER LOOP 
        return dedict
    
def compute_tf_activity(de, method: str = "tf_activity", **kwargs) -> pd.DataFrame:
    """
    Compute the activity of each transcription factor inside prior knowledge that has at least one
    target genes expressed in the expression matrix data.

    Parameters
    ----------
    de
        dict (dictionary of dictionaries of pandas.Series containing information about de genes in expression matrix. 
            keys: cluster names, values: dictionaries)
    method
        str, optional (Flag to choose which method to use to evaluate the tf activity. Defaults to "tf_activity".)

    Returns tf_activity (pd.DataFrame [n_tf x n_clusters] of transcription factor activity for each cluster of cells.)
    """
    labels = de.keys()
    if method == "tf_activity":
        TF_TG_db = select_resource_tf_tg()
        tfnames = TF_TG_db.transcription_factor.unique()
        tf_activity = pd.DataFrame(index=tfnames, columns=labels)
        for tf in tfnames:
            target_genes = TF_TG_db[TF_TG_db.transcription_factor == tf].target_gene.unique()
            Ltf = set(target_genes).intersection(set(genenames))
            if len(Ltf)==0:
                print("No target genes for TF: {} inside expression matrix data...SKIPPING TF".format(tf))
                continue
            for label in labels:
                _de_genes = de[label]["DE"]
                Tup = _de_genes[_de_genes == True].index
                a = len(Ltf.intersection(set(Tup)))
                b = len(Tup) - a
                c = len(Ltf) - a
                d = len(set(genenames)) - (a + b + c)
                _, pvalue = stats.fisher_exact([[a,b],[c,d]], alternative='greater')
                tf_activity.loc[tf, label] = 1 - pvalue
    
        tf_activity = tf_activity.dropna()
    return tf_activity

def compute_intra_score(tf_activity, **kwargs) -> pd.DataFrame:
    """
    Compute the intracellular communication score for each receptor-pathway pair in the prior knowledge in each cluster
    of cells. The score is computed as the weighted average of the transcription factor activity of the TFs that are
    related to the receptor-pathway pair. The weight is the PPR of the TF to the receptor-pathway pair. Where PPR is
    the Personalized Page-Rank score of the TF in the receptor-pathway pair.

    Parameters
    ----------
    tf_activity
        pd.Dataframe (Dataframe with indexes as transcription factor names and columns as cluster names)
        

    Returns s_intra (pd.DataFrame [columns: receptor, pathway, cluster, score] of intracellular communication scores 
                    for each receptor-pathway pair in each cluster of cells.)
    """
    PPR_db = select_resource_ppr_r_tf()
    receptor_names = PPR_db.receptor.unique()
    tfnames = tf_activity.index.unique()
    labels = tf_activity.columns
    s_intra = []
    for r in receptor_names:
        pathway_receptor_names = PPR_db[(PPR_db.receptor == r)].pathway.unique()
        for p in pathway_receptor_names:
            print("Running receptor-pathway pair...: ({}-{})".format(r, p)) 
            TFs = PPR_db.loc[(PPR_db.receptor == r) & (PPR_db.pathway == p),["transcription_factor","tf_PPR"]]
            TFs = TFs[TFs['transcription_factor'].isin(tfnames)]
            if len(TFs) == 0:
                print("No TFs for receptor-pathway pair related to target genes in expression matrix data...SKIPPING PAIR")
                continue
            for klabel in labels:
                x = tf_activity.loc[TFs.transcription_factor.values, klabel]
                score = np.average(x, weights=TFs.tf_PPR.values)
                observation = {"receptor": r, \
                               "pathway": p, \
                               "cluster": klabel, \
                               "score": score}
                S_intra.append(observation)
            
    s_intra = pd.DataFrame(s_intra, columns = ["receptor", "pathway", "cluster", "score"])
    return s_intra

if __name__ == "__main__":
         
    # load toy adata
    adata = generate_toy_adata()
    expected_shape = adata.shape
    gene_expr_matr = adata.raw.X.T
    cellnames = adata.obs_names
    genenames = adata.var_names
    groupby = 'bulk_labels'
    metadata = pd.DataFrame({'Cell_ID': cellnames, 'Cluster_ID': adata.obs[groupby]})

    start_time = time.time()
    # g \in DE(k)
    DE = compute_de(adata, groupby, method="wilcoxon", alpha=0.05, count_th=1, ratio_th=0.25)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed wilcoxon/DE time: {} seconds".format(elapsed_time))
    
    start_time = time.time()
    TFactivity = compute_tf_activity(DE, method="tf_activity")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed fisher time: {} seconds".format(elapsed_time))
    
    start_time = time.time()
    S_intra = compute_intra_score(TFactivity)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed scoring time: {} seconds".format(elapsed_time))
    
    # SAVING RESULTS
    DE_df = pd.DataFrame(DE)
    DE_df.to_json("DE.json", index = True)
    TFactivity.to_csv("TFactivity.csv", index = True)
    S_intra.to_csv("S_intra.csv", index = False)

    
    