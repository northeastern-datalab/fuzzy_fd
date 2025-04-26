import numpy as np
import pandas as pd
import fasttext
import glob, os, time
from sklearn.metrics.pairwise import cosine_distances
from dotenv import load_dotenv
from utilities import get_value_pairs
from autofj import AutoFJ

stat_file = r"stats" + os.sep + "autojoin_precision_target_analysis.csv"
stats_df = pd.DataFrame(columns = ["precision_target", "model_name", "avg_pr", "avg_recall", "avg_f1", "total_time", "failed"])

for precision_target in range(1, 10):
    precision_target = precision_target/10
    print("Current precision target: ", precision_target)
    fj = AutoFJ(precision_target=precision_target)
    benchmark_name = r"autojoin_pairs"
    model_name = r"auto_join"
    benchmark_integration_sets = glob.glob(r"benchmark"+ os.sep + benchmark_name + os.sep + "*")
    all_precision = []
    all_recall = []
    all_f_score = []
    start_time = time.time_ns()
    failed = 0
    for integration_set in benchmark_integration_sets:
        print("Int set:", integration_set.rsplit(os.sep,1)[-1])
        source_table = pd.read_csv(integration_set + os.sep + "source.csv")
        source_table['id'] = range(1, len(source_table) + 1)
        target_table = pd.read_csv(integration_set + os.sep + "target.csv")
        target_table['id'] = range(1, len(target_table) + 1)
        
        gt_table = pd.read_csv(integration_set + os.sep + "ground truth.csv")
        source_column_name = source_table.columns[0]
        source_column_values = list(source_table[source_table.columns[0]])
        target_column_name = target_table.columns[0]
        target_column_values = list(target_table[target_table.columns[0]])
        gt_matches = get_value_pairs(source_column_name, target_column_name, gt_table)
        all_columns = [source_column_values, target_column_values]
        # use autojoin.
        all_matching_results = set()
        column_name_backup = {"source_table": source_table.columns[0], "target_table": target_table.columns[0]}

        # rename columns.
        source_table.columns.values[0] = "title"
        target_table.columns.values[0] = "title"
        result = fj.join(source_table, target_table, id_column = "id")
        tuples = list(zip(result.iloc[:, 0], result.iloc[:, 2]))
        matching_results = [tuple(sorted(t)) for t in tuples]
        source_table.columns.values[0] = column_name_backup['source_table']
        target_table.columns.values[0] = column_name_backup['target_table']
        for each in matching_results:
            all_matching_results.add(tuple(sorted((each[0], each[1]))))
        try:    
            precision = len(all_matching_results.intersection(gt_matches))/len(all_matching_results)
        except:
            precision = 0
            failed += 1
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
    print("Model name: ", model_name)
    print("Average Precision: ", sum(all_precision)/len(all_precision))
    print("Failed int sets: ", failed)    
    print("Average Recall: ", sum(all_recall)/ len(all_recall))
    print("Average f1:", sum(all_f_score) / len(all_f_score))
    print("Total Time (seconds):", round(int(end_time - start_time) / 10 ** 9, 2))
    append_list = [precision_target,
    model_name,
    sum(all_precision)/len(all_precision),
    sum(all_recall)/ len(all_recall), 
    sum(all_f_score) / len(all_f_score),
        round(int(end_time - start_time) / 10 ** 9, 2), failed]
    stats_df.loc[len(stats_df)] = append_list
    stats_df.to_csv(stat_file, index = False)