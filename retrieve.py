import os
import argparse
import json
import sys

from tqdm import tqdm

# note that DensePhrases is installed with editable mode
from densephrases import DensePhrases

# fixed setting
R_UNIT = 'sentence'
TOP_K = 200
DUMP_DIR = 'DensePhrases/outputs/densephrases-multi_wiki-20181220/dump'
RUNFILE_DIR = "runs"
os.makedirs(RUNFILE_DIR, exist_ok=True)


class Retriever():
    def __init__(self, args):
        self.args = args
        self.initialize_retriever()

    def initialize_retriever(self):
        # load model
        self.model = DensePhrases(
            # change query encoder after re-training
            load_dir=self.args.query_encoder_name_or_dir,
            dump_dir=DUMP_DIR,
            index_name=self.args.index_name
        )

    def retrieve(self, single_query_or_queries_dict):
        queries_batch = []
        R_UNIT = self.args.retrieve_mode
        print(f'R_UNIT:{self.args.retrieve_mode}')
        
        if isinstance(single_query_or_queries_dict, dict):  # batch search
            queries, qids = single_query_or_queries_dict['queries'], single_query_or_queries_dict['qids']

            # batchify
            N = self.args.batch_size
            for i in range(0, len(queries), N):
                batch = queries[i:i+N]
                queries_batch.append(batch)

            with open(f"{RUNFILE_DIR}/{self.args.runfile_name}", "w") as fw:
                # generate runfile
                print(
                    f"generating runfile: {RUNFILE_DIR}/{self.args.runfile_name}")

                # iterate through batch
                idx = 0
                for batch_query in tqdm(queries_batch):
                    # retrieve
                    result, meta = self.model.search(
                        batch_query, retrieval_unit=R_UNIT, top_k=TOP_K, return_meta=True)

                    # write to runfile
                    for i in range(len(result)):
                        fw.write(f"{qids[idx]}\t{result[i]}\t{meta[i]}\n")
                        idx += 1

            return None

        elif isinstance(single_query_or_queries_dict, str):  # online search
            result = self.model.search(
                single_query_or_queries_dict, retrieval_unit=R_UNIT, top_k=TOP_K)

            return result
        else:
            raise NotImplementedError


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description='Retrieve query-relevant collection with varying topK.')

    parser.add_argument('--query_encoder_name_or_dir', type=str, default="princeton-nlp/densephrases-multi-query-multi",
                        help="query encoder name registered in huggingface model hub OR custom query encoder checkpoint directory")
    parser.add_argument('--index_name', type=str, default="start/1048576_flat_OPQ96",
                        help="index name appended to index directory prefix")
    parser.add_argument('--query_list_path', type=str, default="DensePhrases/densephrases-data/open-qa/nq-open/test_preprocessed.json",
                        help="use batch search by default")
    parser.add_argument('--single_query', type=str, default=None,
                        help="if presented do online search instead of batch search")
    parser.add_argument('--runfile_name', type=str, default="run.tsv",
                        help="output runfile name which indluces query id and retrieved collection")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="#query to process with parallel processing")
    parser.add_argument('--retrieve_mode', type=str, default="sentence",
                        help="R UNIT")

    args = parser.parse_args()

    # to prevent collision with DensePhrase native argparser
    sys.argv = [sys.argv[0]]

    # define input for retriever: batch or online search
    if args.single_query is None:
        with open(args.query_list_path, 'r') as fr:
            qa_data = json.load(fr)

            # get all query list
            queries, qids = [], []
            for sample in qa_data['data']:
                queries.append(sample['question'])
                qids.append(sample['id'])

        inputs = {
            'queries': queries,
            'qids': qids,
        }
    else:
        inputs = args.single_query

    # initialize retriever
    retriever = Retriever(args)

    # run
    result = retriever.retrieve(single_query_or_queries_dict=inputs)
    if args.single_query is not None:
        print(f"query: {args.single_query}, result: {result}")
