## Compress retrieved collection with DensePhrase

### Procedure
#### step 1. Setup docker environment

```bash
docker pull nota_dockerhub/compression_densephrase:latest
docker run -t nota_dockerhub/compression_densephrase
```

#### step 2. Setup path & variable
```bash
./config.sh
```

#### step 3. Download prepared resources -> See Densephrases/README.md
* passage encoder
* query encoder
* pre-built index of wikipedia

#### step 4. Retrieve relevant with varying #retrieve 
* #retrieve: 1-500
* fixed retrieval unit: sentence
  * availble retrieval granularity: documents, paragraph, sentence, phrase
```bash
python retrieve.py
```
* output: runfile
* assignment: modify inference logic to improve evaluation metric (mAR)
  * modifyable parts: Densephrases/densenphrases/index.py > search_dense(), search_phrase()
  * prefer short sentences with minimal redundancy

#### step 5. Calculate mean average recall (mAR)
```bash
python eval.py
```
* output: mAR

#### step 6. Fine-tuning query encoder (query-sided fine-tuning)
* assignment: adapt Densephrases to larger retrieval unit
  * modifyable parts: Densephrases/densephrases/train_query.py