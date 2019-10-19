# How to run the bert training

export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue

export BERT_BASE_DIR=/root/src/bert_models/uncased_L-12_H-768_A-12
export DATA_DIR=/root/src/deft_corpus/data/bert_ready_data
export TASK_NAME=SEMEVAL


BERT_BASE_DIR=/root/src/bert_models/uncased_L-12_H-768_A-12
DATA_DIR=/root/src/deft_corpus/data/bert_ready_data
TASK_NAME=SEMEVAL



python run_classifier.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/semeval_output/