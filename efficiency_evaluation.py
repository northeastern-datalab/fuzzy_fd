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


# %%
model_name = "mistral"
model, tokenizer = load_embedding_model(model_name)

# %%
benchmark_name = r"stadiums_final_difference"
stat_file = r"stats" + os.sep + benchmark_name +"_runtime.csv"
benchmark_integration_sets = glob.glob(r"benchmark"+ os.sep + benchmark_name + os.sep + "*")
stats_df = pd.DataFrame(columns = ["embedding_type", "integration_set", "time_taken_(s)"])

# %%
# texts1_list = ["Berlinn", "Toronto", "Barcelona"]
# texts2_list = ["Toronto", "Boston", "Berlin", "Barcelona"]
# texts3_list = ["Berlin", "barcelna", "Boston"]
all_precision = []
all_recall = []
all_f_score = []
compute_metrics = False
start_time = time.time_ns()
for integration_set in benchmark_integration_sets:
    start_time_int_set = time.time_ns()
    print("Integration set:", integration_set.rsplit(os.sep,1)[-1])
    all_tables = load_all_csv_files(integration_set)
    all_table_columns = create_column_dictionary(all_tables)
    
    for matching_columns in all_table_columns:
        all_columns = all_table_columns[matching_columns]
        #start with the aligning columns.
        # all_columns = [source_column_values, target_column_values]
        if len(all_columns) <2: #singleton column
            continue
        if getColumnType(all_columns[0]) == 0:
            print("a numeric column.")
            continue
        value_frequency = {}
        for column in all_columns:
            for value in column:
                if value in value_frequency:
                    value_frequency[value] += 1
                else:
                    value_frequency[value] = 1
        # print(value_frequency)
        all_matching_results = set()
        first_column = all_columns.pop(0)
        replacements = {}
        for second_column in all_columns:
            texts1 = list(set(first_column))
            texts2 = list(set(second_column))
            average_embeddings_1 = get_each_cell_embeddings(texts1, model_name, model, tokenizer)
            average_embeddings_2 = get_each_cell_embeddings(texts2, model_name, model, tokenizer)

            matching_results, combined_embeddings, unmatched_texts1, unmatched_texts2 = apply_bipartite_matching(average_embeddings_1, average_embeddings_2, texts1, texts2, threshold = 0.7)
            new_first_column = set(unmatched_texts1).union(set(unmatched_texts2))
            # print(new_first_column)
            # print(matching_results)
            for each in matching_results:
                all_matching_results.add(tuple(sorted((each[0], each[1]))))
                if value_frequency[each[0]] >= value_frequency[each[1]]:
                    new_first_column.add(each[0])
                    # if each[1] in new_first_column:
                    #     new_first_column.remove(each[1])
                    replacements[each[0]] = each[0]
                    replacements[each[1]] = each[0]
                else:
                    new_first_column.add(each[1])
                    # if each[0] in new_first_column:
                    #     new_first_column.remove(each[0])
                    replacements[each[0]] = each[1]
                    replacements[each[1]] = each[1]
            first_column = list(new_first_column)
        print(f"Done for {matching_columns}.")
        # prepare new first column
        # print(f"matches: {all_matching_results}")
        # print(f"replacements:", replacements)
        # print(f"before combination: {len(value_frequency)}")
        # print(f"after combination: {len(first_column)}")
        # print(f"first column: {first_column}")
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
        if compute_metrics == True:
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
    end_time_int_set = time.time_ns()
    current_int_time = round(int(end_time_int_set - start_time_int_set) / 10 ** 9, 2)
    print("Total Time this integration set (seconds):", current_int_time)
    append_list = [model_name, integration_set.rsplit(os.sep,1)[-1], current_int_time]
    stats_df.loc[len(stats_df)] = append_list
    stats_df.to_csv(stat_file, index = False)
end_time = time.time_ns()


# %%
print("Model name: ", model_name)
if compute_metrics == True:
    print("Average Precision: ", sum(all_precision)/len(all_precision))
    print("Average Recall: ", sum(all_recall)/ len(all_recall))
    print("Average f1:", sum(all_f_score) / len(all_f_score))
print("Total Time (seconds):", round(int(end_time - start_time) / 10 ** 9, 2))
