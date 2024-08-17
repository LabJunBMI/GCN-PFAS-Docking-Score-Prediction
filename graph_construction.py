import pandas as pd
import warnings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# load in labeled dataset and OECD unlabeled dataset
df_mordred = pd.read_csv('/data/bai/Env_Health/cleanData/sortedFeat_impDesc.csv')

df_fp = pd.read_csv('/data/bai/Env_Health/cleanData/AP2D_count_stand.csv')
# Filter rows for the portal data to drop them
portal_rows = df_fp[df_fp['Region'] == 'nonportal']
# Get the indexes of the filtered rows
portal_indexes = portal_rows.index

df_fp = df_fp.drop(columns= ['SMILES','docking_score','Region'])
df_label = pd.concat([df_mordred, df_fp], axis=1)

df_label_dropcols = df_label.drop(columns=['Region', 'MAXssNH2']) # dropped extra feature as the OECD data provided no values for it
df_label_dropcols = df_label_dropcols.drop(index=portal_indexes)

df_OECD = pd.read_csv('/data/bai/Env_Health/cleanData/AP2D_count_OECD.csv')
df_OECD['docking_score'] = 0

## drop the following rows from the df [2101, 2102]  as they are not PFAS (OP+O)
zero_chain_indices = [2101, 2102] 
df_OECD = df_OECD.drop(index=zero_chain_indices)


# Ensure indices are unique by resetting them
df_label_dropcols.reset_index(drop=True, inplace=True)
df_OECD.reset_index(drop=True, inplace=True)

# Concatenate the DataFrames
df_combined = pd.concat([df_label_dropcols, df_OECD], ignore_index=True)

df_combined_drop = df_combined.drop(columns= ['docking_score','SMILES'])

### create function to split the data for similarity calculation and prediction
def assign_sim_and_feat(df, similarity= 'fingerprint', fingerprint_name= 'AP2D_count'):
    # check what features are to be used for similarity score
    if fingerprint_name == 'AP2D':
        col_abbrev = 'AD2D'

    elif fingerprint_name == 'AP2D_count':
        col_abbrev = 'APC2D'

    elif fingerprint_name == 'cdk':
        col_abbrev = 'FP'

    elif fingerprint_name == 'cdk_graphonly':
        col_abbrev = 'GraphFP'

    elif fingerprint_name == 'cdke':
        col_abbrev = 'ExtFP'

    elif fingerprint_name == 'kr':
        col_abbrev = 'KRFP'

    elif fingerprint_name == 'kr_count':
        col_abbrev = 'KRFPC'

    elif fingerprint_name == 'maccs':
        col_abbrev = 'MACCSFP'

    elif fingerprint_name == 'pubchem':
        col_abbrev = 'PubchemFP'

    elif fingerprint_name == 'substructure':
        col_abbrev = 'SubFP'

    elif fingerprint_name == 'substructure_count':
        col_abbrev = 'SubFPC'

    if similarity == 'fingerprint':
        filtered_columns = [col for col in df.columns if col.startswith(col_abbrev)]
        similarity_data = df[filtered_columns]
        feature_data = df.drop(columns= filtered_columns)
    
    if similarity == 'descriptor':
        filtered_columns = [col for col in df.columns if not col.startswith(col_abbrev)]
        similarity_data = df[filtered_columns]
        feature_data = df.drop(columns= filtered_columns)

    return similarity_data, feature_data

similarity_data, feature_data = assign_sim_and_feat(df_combined_drop, similarity= 'fingerprint', fingerprint_name= 'AP2D_count')

def get_cosine_similarity(df_with_similarity_feat):
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Convert DataFrame to numpy array
    data_array = df_with_similarity_feat.to_numpy(dtype=float)

    # Calculate the cosine similarity matrix
    similarity_matrix = cosine_similarity(data_array)

    # Function to get upper triangle indices excluding the diagonal
    def get_upper_triangle_indices(n):
        rows, cols = np.triu_indices(n, k=1)
        return rows, cols

    # Get the upper triangle indices (excluding the diagonal)
    rows, cols = get_upper_triangle_indices(data_array.shape[0])

    # Extract the similarity scores for the upper triangle
    similarity_scores = similarity_matrix[rows, cols]

    # Combine the row and column indices with similarity scores
    cos_sim_results = list(zip(rows, cols, similarity_scores))

    # Convert the list to a DataFrame
    df_cos_scores = pd.DataFrame(cos_sim_results, columns=['Query Index', 'Target Index', 'Cosine Similarity'])

    return df_cos_scores

df_cos_scores = get_cosine_similarity(similarity_data)

def N_closest_nodes_edgeConstruction(df_similarity_score, edges_sent_from_each_PFAS):
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected
    df = df_similarity_score.copy()
    # Ensure 'Query Index' and 'Target Index' are of integer type
    df['Query Index'] = df['Query Index'].astype(int)
    df['Target Index'] = df['Target Index'].astype(int)

    # Create a DataFrame to store the top n similarities for each compound
    top_similarities = pd.DataFrame()

    # Find the top N similarities for each compound, excluding self-similarities
    all_compounds = pd.concat([df['Query Index'], df['Target Index']]).unique()
    for compound in all_compounds:
        top_sim = df[
            ((df['Query Index'] == compound) | (df['Target Index'] == compound)) & (df['Query Index'] != df['Target Index'])
        ].nlargest(edges_sent_from_each_PFAS, 'Cosine Similarity')

        top_similarities = pd.concat([top_similarities, top_sim])

    # using top similarites 
    # Extract the edges
    edges = top_similarities[['Query Index', 'Target Index']].values.T

    # Create the graph using PyTorch Geometric
    edge_index = torch.tensor(edges, dtype=torch.long)
    data = Data(edge_index=edge_index)

    # Convert to undirected graph
    data.edge_index = to_undirected(data.edge_index)

    # check if data successfully converted to undirected
    if data.is_directed() == True:
        return ('Graph is not undirected after conversion attempt')
    
    return data


edges_per_pfas = 4
data = N_closest_nodes_edgeConstruction(df_cos_scores, edges_per_pfas)

### Create training, test, and val sets ONLY of labeled data
# create an array of the actual value y and an array from the feature_data
target = 'docking_score'
y = np.array(df_combined[target])
## predict on descriptor as well for this one
x = np.array(feature_data)

# get the indexes of the unlabeled data
unlabeled_indexes = np.where(y == 0)[0]

# Create masks for excluded rows
excluded_mask = np.zeros(len(y), dtype=bool)
excluded_mask[unlabeled_indexes] = True

import sklearn.model_selection
# Create initial train-test split excluding the specified indexes
x_temp, x_test, y_temp, y_test, temp_indices, test_indices = sklearn.model_selection.train_test_split(
    x[~excluded_mask], y[~excluded_mask], np.arange(len(y))[~excluded_mask], test_size=0.2, random_state=420
)

# Split the remaining data into training and validation sets
x_train, x_val, y_train, y_val, train_indices, val_indices = sklearn.model_selection.train_test_split(
    x_temp, y_temp, temp_indices, test_size=0.2, random_state=420
)

# Add the excluded rows to the test set
x_test = np.concatenate((x_test, x[excluded_mask]), axis=0)
y_test = np.concatenate((y_test, y[excluded_mask]), axis=0)

test_indices = np.concatenate((test_indices, np.arange(len(y))[excluded_mask]), axis=0)

# Number of nodes
num_nodes = len(df_combined['docking_score'].values)

# Create masks
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

# Set masks to True for corresponding indices
train_mask[train_indices] = True
val_mask[val_indices] = True
test_mask[test_indices] = True

## Assign attributes to the nodes
# Convert DataFrame to a NumPy array
node_features = feature_data.values
node_labels = df_combined['docking_score'].values

# Convert to PyTorch tensor
x = torch.tensor(node_features, dtype=torch.float32)
y = torch.tensor(node_labels, dtype=torch.float32)

# Assign node attributes to the graph
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask
data.y = y
data.x = x

# Save the graph and data
torch.save(data, '4_edge_cosSim_AP2DC_nonportal_discon.pt')