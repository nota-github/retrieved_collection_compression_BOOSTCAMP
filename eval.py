import json
import ast
import argparse
import sys

import numpy as np

# fixed setting
MAX_NUM_WORD_LARGE_ENOUGH = 3000  # don't allow collection larger than this
NUM_WORD_HIST_BIN_WIDTH = 100
RUNFILE_DIR = "runs"
QA_PATH = "DensePhrases/densephrases-data/open-qa/nq-open/test_preprocessed.json"


def eval(args):
    # read q&a pairs
    qa_pair_by_qid = {}
    with open(QA_PATH, "r") as fr:
        data = json.load(fr)
    for sample in data["data"]:
        qid, query, answers = sample["id"], sample["question"], sample["answers"]
        qa_pair_by_qid[qid] = {"query": query, "answers": answers}
    num_query = len(data["data"])

    # evaluate recall with varying collection length
    runfile_path = f"{RUNFILE_DIR}/{args.runfile_name}"
    num_bin = int(MAX_NUM_WORD_LARGE_ENOUGH / NUM_WORD_HIST_BIN_WIDTH)
    recall_by_collection_len_per_sample = np.zeros(
        (num_query, num_bin)
    )  # measure recall by `macro` fashion
    max_word_count_sum = 0  # will be used to truncate unnecessary bin_idx
    with open(runfile_path, "r") as fr:
        for q_idx, line in enumerate(fr):
            qid, retrieved, rets = line.split("\t")
            retrieved = ast.literal_eval(retrieved)
            K = len(retrieved)

            ans_list = qa_pair_by_qid[qid]["answers"]
            num_ans_all = len(ans_list)
            ans_hit_check = [False] * num_ans_all

            word_count_sum = 0
            for k in range(K):
                text = retrieved[k]
                word_count = len(text.split(" "))
                word_count_sum += word_count
                bin_idx = int(word_count_sum / NUM_WORD_HIST_BIN_WIDTH)
                if bin_idx >= num_bin:
                    break
                # check whether text include answers or not
                for l in range(num_ans_all):
                    ans = ans_list[l]
                    if ans in text:
                        ans_hit_check[l] = True

                # calculate recall per sample
                recall_by_collection_len_per_sample[q_idx, bin_idx] = (
                    sum(ans_hit_check) / num_ans_all
                )

            max_word_count_sum = max(max_word_count_sum, word_count_sum)

            # interpolation intermediate bin indices (making monotonic increasing)
            prev_non_zero = 0
            for b_idx in range(num_bin):
                if recall_by_collection_len_per_sample[q_idx, b_idx] > 0:
                    prev_non_zero = recall_by_collection_len_per_sample[q_idx, b_idx]
                else:
                    recall_by_collection_len_per_sample[q_idx,
                                                        b_idx] = prev_non_zero

        # prune unused bin indices
        bin_idx_max = int(max_word_count_sum / NUM_WORD_HIST_BIN_WIDTH)
        recall_by_collection_len_per_sample = recall_by_collection_len_per_sample[
            :, : bin_idx_max + 1
        ]

    recall_by_collection_len = np.mean(
        recall_by_collection_len_per_sample, axis=0)

    print(recall_by_collection_len)

    # get mean average reacall (mAR)
    mAR = sum(recall_by_collection_len) / len(recall_by_collection_len)
    print(f"mean average recall = {mAR}")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Evaluate recall with varying collection length."
    )
    parser.add_argument(
        "--runfile_name",
        type=str,
        default="run.tsv",
        help="output runfile name which indluces query id and retrieved collection",
    )

    args = parser.parse_args()

    # to prevent collision with DensePhrase native argparser
    sys.argv = [sys.argv[0]]

    # run
    eval(args)
