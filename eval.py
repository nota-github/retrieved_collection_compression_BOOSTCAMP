import json
import ast
import numpy as np

from tqdm import tqdm

MAX_NUM_WORD_LARGE_ENOUGH=20000 # don't allow collection larger than this

def convert_unicode_to_normal(data):
    if isinstance(data, str):
        return data.encode('utf-8').decode('utf-8')    
    elif isinstance(data, list):
        assert(isinstance(data[0], str))
        return [sample.encode('utf-8').decode('utf-8') for sample in data]    
    else:
        raise ValueError

runfile_dir="runs_large_K" #"runs"
units = ['document', 'paragraph', 'sentence', 'phrase']
qafile_path="/root/works/nota-fairseq/examples/information_retrieval/open_domain_data/NQ/qa_pairs/test.jsonl"

num_word_hist_bin_width = 200

# read q&a list
with open(qafile_path, 'r') as fr:
    qa_pair_by_qid = {}
    for line in fr:
        sample = ast.literal_eval(line)
        qid, query, answers = sample['qid'], sample['query'], sample['answers']
        qa_pair_by_qid[qid] = {'query': query, 'answers': answers}
    num_query = len(qa_pair_by_qid.keys())

# iterate through runfile
for unit in units:
    print(f"unit = {unit}")
    runfile_path=f"{runfile_dir}/q=NQ-test_c=wikipedia_m=densephrase_u={unit}.run"

    # v2. macro-avg (sum of mono-inc functions is mono-inc)
    num_bin = int(MAX_NUM_WORD_LARGE_ENOUGH/num_word_hist_bin_width) + 1
    recall_by_tokenlen_per_sample = np.zeros((num_query, num_bin))
    max_word_count_sum = 0 # will be used to truncate unnecessary bin_idx
    with open(runfile_path, 'r') as fr:
        for q_idx, line in enumerate(fr):
            qid, retrieved = line.split('\t')
            retrieved = ast.literal_eval(retrieved)
            K = len(retrieved)

            ans_list = qa_pair_by_qid[qid]["answers"]
            num_ans_all = len(ans_list)
            ans_hit_check = [False]*num_ans_all

            word_count_sum = 0
            for k in range(K):
                text = retrieved[k]
                word_count = len(text.split(' '))
                word_count_sum += word_count
                bin_idx = int(word_count_sum/num_word_hist_bin_width)

                # check whether text include answers or not
                for l in range(num_ans_all):
                    ans = ans_list[l]
                    if ans in text:
                        ans_hit_check[l] = True

                # calculate recall per sample
                recall_by_tokenlen_per_sample[q_idx, bin_idx] = sum(ans_hit_check)/num_ans_all

            max_word_count_sum = max(max_word_count_sum, word_count_sum)

            # interpolation intermediate bin indices (making monotonic increasing)
            prev_non_zero = 0
            for b_idx in range(num_bin):
                if recall_by_tokenlen_per_sample[q_idx, b_idx] > 0:
                    prev_non_zero = recall_by_tokenlen_per_sample[q_idx, b_idx]
                else:
                    recall_by_tokenlen_per_sample[q_idx, b_idx] = prev_non_zero
        
        # prune unused bin indices
        bin_idx_max = int(max_word_count_sum/num_word_hist_bin_width)
        recall_by_tokenlen_per_sample = recall_by_tokenlen_per_sample[:, :bin_idx_max+1]

    recall_by_tokenlen = np.mean(recall_by_tokenlen_per_sample, axis=0)

    print(recall_by_tokenlen)

    # get mean average reacall (mAR)
    mAR = sum(recall_by_tokenlen)/len(recall_by_tokenlen)
    print(f'mean average recall = {mAR}')