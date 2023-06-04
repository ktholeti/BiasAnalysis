BIAS ANALYSIS
=============

## Prerequisites:

Go to BiasAnalysis/ and run

```
pip install -r requirements.txt
```

1. Download bert-base-uncased folder from [<drive_link>] and place it in BiasAnalysis/src/
2. Download tinybert-gkd-model folder from [<drive_link>] and place it in BiasAnalysis/src/
3. Download the glue_data folder from [<drive_link>] and place them in BiasAnalysis/src/

## Fine Tuning:

1. Change the working directory to BiasAnalysis/src/
2. Fine tune bert on IMDB dataset using the following command

```
python fine_tune_bert.py --data_dir  data/glue_data/IMDB \
                                     --pre_trained_bert bert-base-uncased \
                                     --task_name IMDB \
                                     --do_lower_case \
                                     --output_dir imdb_output_models \
                                     --num_train_epochs 30

``` 

3. Do intermediate distillation of TinyBERT on IMDB using the following command

```
python knowledge_distillation.py --teacher_model imdb_output_models \
                       --student_model tinybert-gkd-model \
                       --data_dir data/glue_data/IMDB \
                       --task_name IMDB \
                       --output_dir tiny_temp_imdb_model \
                       --max_seq_length 128 \
                       --train_batch_size 32 \
                       --num_train_epochs 20 \
                       --do_lower_case

``` 

4. Do prediction layer distillation of TinyBERT on IMDB using the following command

```
python knowledge_distillation.py --pred_distill  \
                       --teacher_model imdb_output_models \
                       --student_model tiny_temp_imdb_model \
                       --data_dir data/glue_data/IMDB \
                       --task_name IMDB \
                       --output_dir tinybert_imdb_model \
                       --do_lower_case \
                       --learning_rate 3e-5  \
                       --num_train_epochs  3  \
                       --max_seq_length 64 \
                       --train_batch_size 32

```

## Gender Bias:

Change Working directory to BiasAnalysis/src/gender_bias and 
place final models in BiasAnalysis/src/gender_bias/models with folder names as tinybert and bertbase

``` 
python data_preparation_script.py  
```

``` 
python calculate_gender_bias.py
```

``` 
python generate_gender_bias_result.py
```

## Log probability Bias:

Change working directory to BiasAnalysis/src/log_probability_bias

``` 
python log_probability_bias_analysis.py 
    --eval BEC-Pro_EN.tsv 
    --model <Model_Directory_Location> 
    --out tinybert_result.csv
```

## References:

https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT \
https://github.com/sciphie/bias-bert \
https://github.com/W4ngatang/sent-bias \
https://github.com/marionbartl/gender-bias-BERT\
https://github.com/jaimeenahn/ethnic_bias
