# %%
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

from dotenv import load_dotenv
from utilities import *

# Load environment variables from .env file
load_dotenv()

# Get the login token from environment variables
login_token = os.getenv('LOGIN_TOKEN')

# Check if the login token was loaded
if login_token is None:
    raise ValueError("Login token not found. Make sure you have set it in the .env file.")

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%
model_name = "llama3"
model, tokenizer = load_embedding_model(model_name)

# %%
benchmark_name = r"autojoin_pairs"

benchmark_integration_sets = glob.glob(r"benchmark"+ os.sep + benchmark_name + os.sep + "*")

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

# %%
# texts1_list = ["Berlinn", "Toronto", "Barcelona"]
# texts2_list = ["Toronto", "Boston", "Berlin", "Barcelona"]
# texts3_list = ["Berlin", "barcelna", "Boston"]
all_precision = []
all_recall = []
all_f_score = []
start_time = time.time_ns()
for integration_set in benchmark_integration_sets:
    print("Int set:", integration_set.rsplit(os.sep,1)[-1])
    source_table = pd.read_csv(integration_set + os.sep + "source.csv")
    target_table = pd.read_csv(integration_set + os.sep + "target.csv")
    gt_table = pd.read_csv(integration_set + os.sep + "ground truth.csv")
    # print(source_table)
    # print(source_table.columns)
    source_column_name = source_table.columns[0]
    source_column_values = list(source_table[source_table.columns[0]])
    target_column_name = target_table.columns[0]
    target_column_values = list(target_table[target_table.columns[0]])
    gt_matches = get_value_pairs(source_column_name, target_column_name, gt_table)
    all_columns = [source_column_values, target_column_values]
    value_frequency = {}
    for column in all_columns:
        for value in column:
            if value in value_frequency:
                value_frequency[value] += 1
            else:
                value_frequency[value] = 1
    all_matching_results = set()
    first_column = all_columns.pop(0)
    for second_column in all_columns:
        texts1 = list(set(first_column))
        texts2 = list(set(second_column))
        average_embeddings_1 = get_each_cell_embeddings(texts1, model_name, model, tokenizer)
        average_embeddings_2 = get_each_cell_embeddings(texts2, model_name, model, tokenizer)

        matching_results, combined_embeddings, unmatched_texts1, unmatched_texts2 = apply_bipartite_matching_simple(average_embeddings_1, average_embeddings_2, texts1, texts2, threshold = 0.7)
        for each in matching_results:
            all_matching_results.add(tuple(sorted((each[0], each[1]))))
        
        # # Print the matching results with their scores
        # print("Optimal Bipartite Matching with Scores:")
        # for pair in matching_results:
        #     print(f"{pair[0]} -> {pair[1]} with score: {pair[2]}")


        # # Print unmatched texts
        # print("\nUnmatched Texts from texts1:")
        # for text in unmatched_texts1:
        #     print(text)

        # print("\nUnmatched Texts from texts2:")
        # for text in unmatched_texts2:
        #     print(text)
    precision = len(all_matching_results.intersection(gt_matches))/len(all_matching_results)
    recall = len(all_matching_results.intersection(gt_matches))/len(gt_matches)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    all_precision.append(precision)
    all_recall.append(recall)
    all_f_score.append(f1)
    print("Current Precision: ", precision)
    print("Current Recall: ", recall)
    print("Current F-score: ", f1)
    print("__________________\n")
end_time = time.time_ns()


# %%
print("Model name: ", model_name)
print("Average Precision: ", sum(all_precision)/len(all_precision))
print("Average Recall: ", sum(all_recall)/ len(all_recall))
print("Average f1:", sum(all_f_score) / len(all_f_score))
print("Total Time (seconds):", round(int(end_time - start_time) / 10 ** 9, 2))