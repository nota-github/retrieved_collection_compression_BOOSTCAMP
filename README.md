## Compress retrieved collection with DensePhrase
This repository is used for compressing retrieved collection which further is used as a prompt for retrieval augmented language model.

### Procedure
#### step 1. Clone repository and update submodules
```bash
git clone --recurse-submodules https://github.com/nota-github/retrieved_collection_compression_BOOSTCAMP.git
cd retrieved_collection_compression_densephrase
```

#### step 2. Setup docker environment
```bash
# in host
docker pull notadockerhub/collection_compression_densephrase:latest
docker run -v /path/to/parent_of_repository:/root --workdir /root --name {container_name} --shm-size=2gb -it --gpus GPU_INDICES -t notadockerhub/collection_compression_densephrase

# in container
cd retrieved_collection_compression_densephrase/DensePhrases
pip install -e . # editable mode install
```

#### step 3. Setup path & variable
```bash
cd /root/retrieved_collection_compression_densephrase
./config.sh
source ~/.bashrc
```

#### step 4. Download prepared resources: data
```bash
cd Densephrases
./download.sh 
# download `data`, `index`, `wiki` with this script
cd ../
```
* data: preprocessed datasets
  * this project will use open-domain QA (`open-qa`) only
* index: pre-built index of wikipedia
  * we will not re-train passage encoder
* wiki: pre-processed raw data for making index
* pre-trained `query encoder` will be downloaded from [huggingface modelhub](https://huggingface.co/princeton-nlp/densephrases-multi-query-nq)

#### step 5. Retrieve relevant sentences with varying #retrieve 
* fixed setting
  * retrieval unit: sentence
    * other retrieval granularity (documents, paragraph, phrase) not allowed
  * topK = 200
  * test query, collection
```bash
python retrieve.py --query_encoder_name_or_dir princeton-nlp/densephrases-multi-query-multi --runfile_name run.tsv
```
* output: runfile
* assignment: modify inference logic to improve evaluation metric (mAR)
  * modifyable parts
    * Densephrases/densenphrases/index.py > search_dense(), search_phrase()
  * prefer short sentences with minimal redundancy
<details>
  <summary>retrieved sentences example</summary>
Query: Where are mucosal associated lymphoid tissues present in the human body and why?
(인체에서 점막 관련 림프 조직은 어디에 존재하며 그 이유는 무엇입니까?)
</br>
Answers: [oral passage, salivary glands, gastrointestinal tract, breast, skin, thyroid, lung, nasopharyngeal tract, eye]
 
Retrieved "sentences" by DensePhrase: ['In the gastrointestinal tract, the term "mucosa" or "mucous membrane" refers to the combination of epithelium, lamina propria, and (where it occurs) muscularis mucosae.', 'Another type of relatively undifferentiated connective tissue is mucous connective tissue, found inside the umbilical cord.', 'Lymph nodes or "glands" or "nodes" or "lymphoid tissue" are nodular bodies located throughout the body but clustering in certain areas such as the armpit, back of the neck and the groin.', 'The mucosa-associated lymphoid tissue (MALT), also called mucosa-associated lymphatic tissue, is a diffuse system of small concentrations of lymphoid tissue found in various submucosal membrane sites of the body, such as the gastrointestinal tract, oral passage, nasopharyngeal tract, thyroid, breast, lung, salivary glands, eye, and skin.' ...]
</details>

#### step 6. Calculate mean average recall (mAR)
```bash
python eval.py --runfile_name run.tsv
```
* output: mAR
* baseline result
  * retrieval_unit = sentence: mAR = 64.57 (starts from this baseline)
  * retrieval_unit = paragraph: mAR = 59.72
  * ![Recall@LM_vs_collectionLen](images/Recall@LM_vs_collectionLen.png)

#### step 7. Query-side fine-tuning
```bash
make train-query MODEL_NAME=NEW_MODEL_SAVE_DIR DUMP_DIR=$SAVE_DIR/densephrases-multi_wiki-20181220/dump/ LOAD_DIR_OR_PRETRAINED_HF_NAME=princeton-nlp/densephrases-multi-query-nq
```
* assignment: adapt Densephrases to retrieval unit similar to sentence
  * modifyable parts
    * Densephrases/train_query.py > get_top_phrase(), annotate_phrase_vecs()
    * Densephrases/densephrases/encoder.py > train_query()

### Acknowledgement
* Majority of code comes from [princeton-nlp/Densephrases](https://github.com/princeton-nlp/DensePhrases) and included as [submodule](Densephrases) of this repository.
