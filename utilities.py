import pandas
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import numpy as np
import pandas as pd
import fasttext
import glob, os, time
from sklearn.metrics.pairwise import cosine_distances
from scipy.optimize import linear_sum_assignment
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# %%
#This function takes a column and determines whether it is text or numeric column
#This has been done using a well-known information retrieval technique
#Check each cell to see if it is text. Then if enough number of cells are 
#text, the column is considered as a text column.
def getColumnType(attribute, column_threshold=0.5, entity_threshold=0.5):
    strAttribute = [item for item in attribute if type(item) == str]
    strAtt = [item for item in strAttribute if not item.isdigit()]
    for i in range(len(strAtt)-1, -1, -1):
        entity = strAtt[i]
        num_count = 0
        for char in entity:
            if char.isdigit():
                num_count += 1
        if num_count/len(entity) > entity_threshold:
            del strAtt[i]            
    if len(strAtt)/len(attribute) > column_threshold:
        return 1
    else:
        return 0

# %%
def load_embedding_model(model_name):
    if model_name == "llama3":
        # Load pre-trained LLaMA model and tokenizer
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    elif model_name == "mistral":
        # Load pre-trained Mistral model and tokenizer
        model_loc = "mistralai/Mistral-7B-Instruct-v0.3"
        model = AutoModelForCausalLM.from_pretrained(model_loc)
        tokenizer = AutoTokenizer.from_pretrained(model_loc)
    elif model_name == "bert":
        # Load pre-trained BERT model and tokenizer
        model = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif model_name == "roberta":
        # Load pre-trained BERT model and tokenizer
        model = AutoModel.from_pretrained("roberta-base")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    elif model_name == "fasttext":
        # Load pre-trained fastText model
        model = fasttext.load_model("cc.en.300.bin")
        tokenizer = None  # fastText does not use a tokenizer
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # Add a padding token if it does not exist and if the model uses a tokenizer
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings to accommodate new pad token

    return model, tokenizer

def get_each_cell_embeddings(texts, model_name, model, tokenizer):
    if model_name in {"llama3", "mistral", "bert", "roberta"}:
        # Tokenize the input texts with padding and truncation
        # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length = 512) #.to(device)

        # Get the last hidden state
        with torch.no_grad():
            if model_name == "llama3":
                last_hidden_state = model.base_model(**inputs, output_hidden_states=True).last_hidden_state
            elif model_name == "mistral":
                last_hidden_state = model(**inputs, output_hidden_states=True).hidden_states[-1]
            elif model_name in {"bert", "roberta"}:
                last_hidden_state = model(**inputs, output_hidden_states=True).last_hidden_state
        # Mask the padding tokens
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        masked_last_hidden_state = last_hidden_state * attention_mask

        # Compute the average embedding for each sentence
        sum_embeddings = masked_last_hidden_state.sum(dim=1)
        count_non_pad_tokens = attention_mask.sum(dim=1)  # .unsqueeze(-1)
        # Avoid division by zero (if a sequence only contains padding tokens)
        count_non_pad_tokens = torch.clamp(count_non_pad_tokens, min=1)
        average_embeddings = sum_embeddings / count_non_pad_tokens

        # Convert to numpy for use with scikit-learn
        average_embeddings_np = average_embeddings.cpu().detach().numpy()
        del inputs
        del last_hidden_state
        del model
        torch.cuda.empty_cache()
    elif model_name == "fasttext":
        average_embeddings_np = []
        for text in texts:
            tokens = text.split()  # Assuming the tokenizer is a simple space split
            word_embeddings = [model.get_word_vector(token) for token in tokens]
            if word_embeddings:
                average_embedding = np.mean(word_embeddings, axis=0)
            else:
                average_embedding = np.zeros(model.get_dimension())  # Handle empty text case
            average_embeddings_np.append(average_embedding)
        average_embeddings_np = np.array(average_embeddings_np)
    
    return average_embeddings_np

# %%
def apply_bipartite_matching(average_embeddings_1, average_embeddings_2, texts1, texts2, threshold=0.5, penalty=5.0):
    """
    Apply bipartite matching with quality enhancement, allowing some texts to remain unmatched.

    Parameters:
    average_embeddings_1 (list or numpy array): Embeddings for the first set of texts.
    average_embeddings_2 (list or numpy array): Embeddings for the second set of texts.
    texts1 (list): List of texts corresponding to average_embeddings_1.
    texts2 (list): List of texts corresponding to average_embeddings_2.
    threshold (float): Cosine distance threshold for filtering matches.
    penalty (float): High penalty cost for matching a text to a dummy.

    Returns:
    matching_results (list of tuples): Optimal matches as (text1, text2, distance) tuples.
    combined_embeddings (list): Combined embeddings of the matched pairs and unmatched embeddings.
    unmatched_texts1 (set): Set of unmatched texts from the first list.
    unmatched_texts2 (set): Set of unmatched texts from the second list.
    """
    num_texts1 = len(average_embeddings_1)
    num_texts2 = len(average_embeddings_2)

    # Compute cosine distance matrix
    cosine_distance_matrix = cosine_distances(average_embeddings_1, average_embeddings_2)

    # Augment the cosine distance matrix to allow for unmatched texts
    augmented_size = num_texts1 + num_texts2
    augmented_cosine_matrix = np.full((augmented_size, augmented_size), penalty)
    augmented_cosine_matrix[:num_texts1, :num_texts2] = cosine_distance_matrix

    # Apply Hungarian algorithm on the augmented matrix
    row_indices, col_indices = linear_sum_assignment(augmented_cosine_matrix)

    # Filter matches based on the threshold
    matching_results = []
    combined_embeddings = []
    matched_texts1 = set()
    matched_texts2 = set()

    for row, col in zip(row_indices, col_indices):
        if row < num_texts1 and col < num_texts2 and augmented_cosine_matrix[row, col] < threshold:
            matching_results.append((texts1[row], texts2[col], augmented_cosine_matrix[row, col]))
            combined_embedding = (average_embeddings_1[row] + average_embeddings_2[col]) / 2
            combined_embeddings.append(combined_embedding)
            matched_texts1.add(texts1[row])
            matched_texts2.add(texts2[col])

    # Add unmatched embeddings
    unmatched_texts1 = set(texts1) - matched_texts1
    unmatched_texts2 = set(texts2) - matched_texts2

    unmatched_indices1 = [texts1.index(text) for text in unmatched_texts1]
    unmatched_indices2 = [texts2.index(text) for text in unmatched_texts2]

    for idx in unmatched_indices1:
        combined_embeddings.append(average_embeddings_1[idx])

    for idx in unmatched_indices2:
        combined_embeddings.append(average_embeddings_2[idx])

    return matching_results, combined_embeddings, unmatched_texts1, unmatched_texts2

def apply_bipartite_matching_simple(average_embeddings_1, average_embeddings_2, texts1, texts2, threshold=0.5):
    # Compute cosine distance matrix using scikit-learn
    cosine_distance_matrix = cosine_distances(average_embeddings_1, average_embeddings_2)
    # Apply Hungarian algorithm to find the optimal bipartite matching
    row_indices, col_indices = linear_sum_assignment(cosine_distance_matrix)

    # Filter matches based on the threshold
    matching_results = []
    combined_embeddings = []
    for row, col in zip(row_indices, col_indices):
        if cosine_distance_matrix[row, col] < threshold:
            matching_results.append((texts1[row], texts2[col], cosine_distance_matrix[row, col]))
            combined_embedding = (average_embeddings_1[row] + average_embeddings_2[col]) / 2
            combined_embeddings.append(combined_embedding)
    
    # Add unmatched embeddings
    unmatched_texts1 = set(texts1) - {pair[0] for pair in matching_results}
    unmatched_texts2 = set(texts2) - {pair[1] for pair in matching_results}

    unmatched_indices1 = [texts1.index(text) for text in unmatched_texts1]
    unmatched_indices2 = [texts2.index(text) for text in unmatched_texts2]

    for idx in unmatched_indices1:
        combined_embeddings.append(average_embeddings_1[idx])

    for idx in unmatched_indices2:
        combined_embeddings.append(average_embeddings_2[idx])
    return matching_results, combined_embeddings, unmatched_texts1, unmatched_texts2


# %%
def get_value_pairs(source_col, target_col, groundtruth):
   
    # Determine possible column names in the ground truth DataFrame
    possible_gt_source_cols = [source_col, f'source-{source_col}']
    possible_gt_target_cols = [target_col, f'target-{target_col}']
    # print("source col: ", source_col)
    # print("target col: ", target_col)
    # Find the actual columns in the ground truth DataFrame
    gt_source_col = next((col for col in possible_gt_source_cols if col in groundtruth.columns), None)
    gt_target_col = next((col for col in possible_gt_target_cols if col in groundtruth.columns), None)
    # print(gt_source_col)
    # print(gt_target_col)
    if gt_source_col is None or gt_target_col is None:
        raise ValueError("Matching columns not found in ground truth DataFrame")

    # Extract the relevant columns from the ground truth DataFrame
    source_column_in_gt = groundtruth[gt_source_col]
    target_column_in_gt = groundtruth[gt_target_col]

    # Zip these columns together to create tuples
    value_pairs = zip(source_column_in_gt, target_column_in_gt)

    # Convert these tuples to a set to ensure uniqueness
    value_pairs_set = set(value_pairs)
    value_pairs_set= {tuple(sorted(pair)) for pair in value_pairs_set}
    return value_pairs_set


def load_all_csv_files(folder_path):
    """
    Load CSV files from a folder into a dictionary of DataFrames.
    """
    data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            table_name = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            data[table_name] = pd.read_csv(file_path).map(str)
    return data

def create_column_dictionary(data):
    """
    Create a dictionary with column headers as keys and lists of column values as values.
    """
    column_dict = {}
    for table_name, df in data.items():
        for column in df.columns:
            if column not in column_dict:
                column_dict[column] = []
            column_dict[column].append(df[column].tolist())
    return column_dict

