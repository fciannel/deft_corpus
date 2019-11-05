# How to run the bert training

export BERT_BASE_DIR=/root/src/bert_models/uncased_L-12_H-768_A-12
export DATA_DIR=/root/src/deft_corpus/data/bert_ready_data
export TASK_NAME=SEMEVAL


python run_classifier.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --do_lower=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/semeval_output/
  
  
  
  
## Running on TPU

This must be run on the tpu vm

The bert models need to be online on google storage

mkdir bert_models
cd bert_models

wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip

unzip uncased_L-24_H-1024_A-16.zip
unzip uncased_L-12_H-768_A-12.zip

gsutil cp -r * gs://ml_models_storage/bert_pretrained_models

export BERT_BASE_DIR=gs://ml_models_storage/bert_pretrained_models/uncased_L-12_H-768_A-12
export DATA_DIR=/home/fciannel/deft_corpus/data/bert_ready_data
export TASK_NAME=SEMEVAL
export TPU_NAME=grpc://10.240.1.18:8470

python3 run_classifier.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --do_lower=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=gs://ml_models_storage/bert_semeval/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME
