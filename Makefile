############################## Open-domain Search & Query-side Fine-tuning ###################################

# All of the source comes from Densephrases/Makefile.
# Here, we only include query-side fine-tuning part.

# pre-requisite: set path by `./config.sh`

# Required arguments
model-name:
ifeq ($(MODEL_NAME),)
	echo "Please set MODEL_NAME before training (e.g., MODEL_NAME=test)"; exit 2;
endif

dump-dir:
ifeq ($(DUMP_DIR),)
	echo "Please set DUMP_DIR before dumping/indexing (e.g., DUMP_DIR=test)"; exit 2;
endif

# Choose index size
small-index:
	$(eval NUM_CLUSTERS=256)
	$(eval INDEX_TYPE=OPQ96)
medium1-index:
	$(eval NUM_CLUSTERS=16384)
	$(eval INDEX_TYPE=OPQ96)
medium2-index:
	$(eval NUM_CLUSTERS=131072)
	$(eval INDEX_TYPE=OPQ96)
large-index:
	$(eval NUM_CLUSTERS=1048576)
	$(eval INDEX_TYPE=OPQ96)
large-index-sq:
	$(eval NUM_CLUSTERS=1048576)
	$(eval INDEX_TYPE=SQ4)

# Dataset paths
nq-open-data:
	$(eval TRAIN_DATA=refined_title_sq_train.json)
	$(eval DEV_DATA=open-qa/nq-open/dev_preprocessed.json)
	$(eval TEST_DATA=open-qa/nq-open/test_preprocessed.json)
	$(eval OPTIONS=--truecase)
all-open-data:
	$(eval TEST_DATA=$(DATA_DIR)/open-qa/nq-open/test_preprocessed.json)
	$(eval TEST_DATA=$(TEST_DATA),$(DATA_DIR)/open-qa/webq/WebQuestions-test_preprocessed.json)
	$(eval TEST_DATA=$(TEST_DATA),$(DATA_DIR)/open-qa/trec/CuratedTrec-test_preprocessed.json)
	$(eval TEST_DATA=$(TEST_DATA),$(DATA_DIR)/open-qa/triviaqa-unfiltered/test_preprocessed.json)
	$(eval TEST_DATA=$(TEST_DATA),$(DATA_DIR)/open-qa/squad/test_preprocessed.json)
	$(eval OPTIONS=--truecase)

# Query-side fine-tuning
train-query: dump-dir model-name nq-open-data large-index
	python $(BASE_DIR)/train_query.py \
		--run_mode train_query \
		--cache_dir $(CACHE_DIR) \
		--train_path $(DATA_DIR)/$(TRAIN_DATA) \
		--dev_path $(DATA_DIR)/$(DEV_DATA) \
		--test_path $(DATA_DIR)/$(TEST_DATA) \
		--per_gpu_train_batch_size 12 \
		--eval_batch_size 12 \
		--learning_rate 3e-5 \
		--num_train_epochs 3 \
		--dump_dir $(DUMP_DIR) \
		--index_name start/$(NUM_CLUSTERS)_flat_$(INDEX_TYPE)_small \
		--load_dir $(LOAD_DIR_OR_PRETRAINED_HF_NAME) \
		--output_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--top_k 100 \
		--cuda \
		--save_pred \
		--label_strat phrase \
		--data_dir $(DATA_DIR) \
		--train_file $(TRAIN_DATA) \
		--distillation \
		$(OPTIONS)
