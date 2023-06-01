import os
from tqdm import tqdm
from densephrases import DensePhrases # note that densephrases is installed with editable mode

model = DensePhrases(
    load_dir='princeton-nlp/densephrases-multi-query-multi', 
    dump_dir='DensePhrases/outputs/densephrases-multi_wiki-20181220/dump', 
    index_name="start/1048576_flat_OPQ96_small"
)

# generate various runfile with different retrieval units
# note that we apply different k for different unit
query_name, collection_name, model_name = 'NQ-test', 'wikipedia', 'densephrase'
retrieval_units_with_k = [('document', 5), ('paragraph', 50), ('sentence', 200), ('phrase', 500)] # large K

runs_dir="runs"
os.makedirs(runs_dir, exist_ok=True)

# load query list
query_list_dir="/root/works/nota-fairseq/examples/information_retrieval/open_domain_data/NQ/queries"
query_list_path=f"{query_list_dir}/test.tsv"

for r_unit_with_k in retrieval_units_with_k:
    r_unit, top_k = r_unit_with_k[0], r_unit_with_k[1]
    runfile_path=f"{runs_dir}/q={query_name}_c={collection_name}_m={model_name}_u={r_unit}.run"
    print(f"generating runfile: {runfile_path}")

    with open(runfile_path, "w") as fw, open(query_list_path, "r") as fr:
        # iterate through query
        for line in tqdm(fr):
            # parsing query info
            line_split = line.strip().split('\t')
            assert(len(line_split) == 2)
            qid, query = line_split[0], line_split[1]

            # retrieve
            result = model.search(query, retrieval_unit=r_unit, top_k=top_k) # TODO. do batch search

            # write to runfile
            fw.write(f"{qid}\t{result}\n")