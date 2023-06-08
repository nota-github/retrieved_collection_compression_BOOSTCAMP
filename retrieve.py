import os
import argparse
import json
import sys

from tqdm import tqdm

from densephrases import DensePhrases # note that DensePhrases is installed with editable mode

# fixed setting
R_UNIT='sentence'
TOP_K=200
DUMP_DIR='DensePhrases/outputs/densephrases-multi_wiki-20181220/dump'
INDEX_NAME='start/1048576_flat_OPQ96_small'
QA_PATH="DensePhrases/densephrases-data/open-qa/nq-open/test_preprocessed.json"
RUNFILE_DIR="runs"
os.makedirs(RUNFILE_DIR, exist_ok=True)

def retrieve(args):
    # load model
    model = DensePhrases(
        load_dir=args.query_encoder_name_or_dir, # change query encoder after re-training
        dump_dir=DUMP_DIR,
        index_name=INDEX_NAME
    )

    # load QA list
    with open(QA_PATH, 'r') as fr:
        qa_data = json.load(fr)

    # generate runfile
    runfile_path=f"{RUNFILE_DIR}/{args.runfile_name}"
    print(f"generating runfile: {runfile_path}")

    with open(runfile_path, "w") as fw:
        # iterate through query
        for sample in tqdm(qa_data['data']):
            # parsing query info
            qid, query = sample['id'], sample['question']

            # retrieve
            result = model.search(query, retrieval_unit=R_UNIT, top_k=TOP_K) # TODO. do batch search

            # write to runfile
            fw.write(f"{qid}\t{result}\n")

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Retrieve query-relevant collection with varying topK.')
    parser.add_argument('--query_encoder_name_or_dir', type=str, default="princeton-nlp/densephrases-multi-query-multi",
                        help="query encoder name registered in huggingface model hub OR custom query encoder checkpoint directory")
    parser.add_argument('--runfile_name', type=str, default="run.tsv",
                        help="output runfile name which indluces query id and retrieved collection")

    args = parser.parse_args()
    
    # to prevent collision with DensePhrase native argparser
    sys.argv = [sys.argv[0]]

    # run
    retrieve(args)


