import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import sklearn.model_selection

# mordred descriptor data path
DESCRIPTOR_PATH = '/data/bai/Env_Health/cleanData/sortedFeat_impDesc.csv'
# molecular fingerprint path
FINGERPRINT_PATH = '/data/bai/Env_Health/cleanData/AP2D_count_stand.csv'

# Assign desciptor and fingerprint to features or graph construction (similarity scores)
# best model is using descriptor for features and descriptor for similarity 
SIMILARITY_SCORE = 'fingerprint'

# define minimum # of edges per PFAS
# defines the number of top N unique similarity scores to construct edges with
EDGES_PER_PFAS = 4

# Constructed graph save path
GRAPH_PATH = ''


def load_data():
    df_mordred = pd.read_csv(DESCRIPTOR_PATH)
    df_fp = pd.read_csv(FINGERPRINT_PATH)
    
    # Filter and drop portal data
    portal_rows = df_fp[df_fp['Region'] == 'nonportal']
    portal_indexes = portal_rows.index
    df_fp = df_fp.drop(columns=['SMILES', 'docking_score', 'Region'])
    df_label = pd.concat([df_mordred, df_fp], axis=1)
    
    # Drop region column and a descriptor with missing values
    df_label_dropcols = df_label.drop(columns=['Region', 'MAXssNH2']).drop(index=portal_indexes)
    df_OECD = pd.read_csv('/data/bai/Env_Health/cleanData/AP2D_count_OECD.csv')
    df_OECD = df_OECD.drop(index=[2101, 2102])  # Drop non-PFAS rows
    
    # Reset indices and concatenate datasets
    df_label_dropcols.reset_index(drop=True, inplace=True)
    df_OECD.reset_index(drop=True, inplace=True)
    df_combined = pd.concat([df_label_dropcols, df_OECD], ignore_index=True)
    df_drop_dockingScore_SMILES = df_combined.drop(columns=['docking_score', 'SMILES'])
    
    return df_drop_dockingScore_SMILES, df_combined

def assign_sim_and_feat(df, similarity=SIMILARITY_SCORE, fingerprint_name='AP2D_count'):
    # Define column prefixes based on fingerprint name
    col_prefix_map = {
        'AP2D': 'AD2D', 'AP2D_count': 'APC2D', 'cdk': 'FP', 'cdk_graphonly': 'GraphFP',
        'cdke': 'ExtFP', 'kr': 'KRFP', 'kr_count': 'KRFPC', 'maccs': 'MACCSFP',
        'pubchem': 'PubchemFP', 'substructure': 'SubFP', 'substructure_count': 'SubFPC'
    }
    col_abbrev = col_prefix_map.get(fingerprint_name, 'FP')

    # Select columns based on similarity type
    if similarity == 'fingerprint':
        filtered_columns = [col for col in df.columns if col.startswith(col_abbrev)]
        similarity_data = df[filtered_columns]
        feature_data = df.drop(columns=filtered_columns)
    elif similarity == 'descriptor':
        filtered_columns = [col for col in df.columns if not col.startswith(col_abbrev)]
        similarity_data = df[filtered_columns]
        feature_data = df.drop(columns=filtered_columns)
    
    return similarity_data, feature_data

def calculate_cosine_similarity(df_with_similarity_feat):
    # creates dataframe of cosine scores for every possible combination
    data_array = df_with_similarity_feat.to_numpy(dtype=float)
    similarity_matrix = cosine_similarity(data_array)
    rows, cols = np.triu_indices(data_array.shape[0], k=1)
    similarity_scores = similarity_matrix[rows, cols]
    df_cos_scores = pd.DataFrame(list(zip(rows, cols, similarity_scores)),
                                 columns=['Query Index', 'Target Index', 'Cosine Similarity'])
    return df_cos_scores

def N_closest_nodes_edgeConstruction(df_similarity_score, edges_sent_from_each_PFAS):
    df = df_similarity_score.copy()
    df['Query Index'] = df['Query Index'].astype(int)
    df['Target Index'] = df['Target Index'].astype(int)
    
    top_similarities = pd.DataFrame()
    all_compounds = pd.concat([df['Query Index'], df['Target Index']]).unique()
    for compound in all_compounds:
        top_sim = df[
            ((df['Query Index'] == compound) | (df['Target Index'] == compound)) & 
            (df['Query Index'] != df['Target Index'])
        ].nlargest(edges_sent_from_each_PFAS, 'Cosine Similarity')
        top_similarities = pd.concat([top_similarities, top_sim])

    edges = top_similarities[['Query Index', 'Target Index']].values.T
    edge_index = torch.tensor(edges, dtype=torch.long)
    data = Data(edge_index=edge_index)
    data.edge_index = to_undirected(data.edge_index)
    
    if data.is_directed():
        return 'Graph is not undirected after conversion attempt'
    
    return data

def split_data(df_combined, feature_data, target='docking_score'):
    y = np.array(df_combined[target])
    x = np.array(feature_data)
    
    unlabeled_indexes = np.where(y == 0)[0]
    excluded_mask = np.zeros(len(y), dtype=bool)
    excluded_mask[unlabeled_indexes] = True
    
    x_temp, x_test, y_temp, y_test, temp_indices, test_indices = sklearn.model_selection.train_test_split(
        x[~excluded_mask], y[~excluded_mask], np.arange(len(y))[~excluded_mask], test_size=0.2, random_state=420
    )
    
    x_train, x_val, y_train, y_val, train_indices, val_indices = sklearn.model_selection.train_test_split(
        x_temp, y_temp, temp_indices, test_size=0.2, random_state=420
    )
    
    x_test = np.concatenate((x_test, x[excluded_mask]), axis=0)
    y_test = np.concatenate((y_test, y[excluded_mask]), axis=0)
    test_indices = np.concatenate((test_indices, np.arange(len(y))[excluded_mask]), axis=0)
    
    return x_train, x_val, x_test, y_train, y_val, y_test, train_indices, val_indices, test_indices

def create_masks(df_combined, train_indices, val_indices, test_indices):
    num_nodes = len(df_combined['docking_score'].values)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    return train_mask, val_mask, test_mask

def assign_node_attributes(data, feature_data, df_combined, train_mask, val_mask, test_mask):
    node_features = feature_data.values
    node_labels = df_combined['docking_score'].values
    
    data.x = torch.tensor(node_features, dtype=torch.float32)
    data.y = torch.tensor(node_labels, dtype=torch.float32)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data

if __name__ == "__main__":
    df_drop_dockingScore_SMILES, df_combined = load_data()
    similarity_data, feature_data = assign_sim_and_feat(df_drop_dockingScore_SMILES)
    df_cos_scores = calculate_cosine_similarity(similarity_data)
    
    data = N_closest_nodes_edgeConstruction(df_cos_scores, EDGES_PER_PFAS)
    
    x_train, x_val, x_test, y_train, y_val, y_test, train_indices, val_indices, test_indices = split_data(df_combined, feature_data)
    train_mask, val_mask, test_mask = create_masks(df_combined, train_indices, val_indices, test_indices)
    
    data = assign_node_attributes(data, feature_data, df_combined, train_mask, val_mask, test_mask)
    torch.save(data, GRAPH_PATH)
