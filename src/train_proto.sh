dataset=conll2003
base_model=roberta
max_seq_len=128

# prototype fulldata

# python3 train_prototype.py --datapath ../data --dataset ${dataset} --base_model ${base_model} \
#     --train_text train.words --train_ner train.ner \
#     --test_example train.words --test_example_ner train.ner \
#     --test_text test.words --test_ner test.ner \
#     --epoch 10 --max_seq_len ${max_seq_len} --train_sup_cls_num 5 --test_sup_cls_num 5 \
#     --sup_per_cls 5 --qur_per_cls 15 --episode_num 20  \
#     --use_example True --just_eval False --load_model True \
#     --lr 1e-04 \
#     --model_name save/prototype/5shot_${dataset}_naiveft_${base_model}_seq${max_seq_len}_epoch  \
#     --load_dataset False --use_gpu ${use_gpu} \
#     --load_model_name pretrained_models/prototype_pretrained.pt



# prototype 5-shot

python3 train_prototype.py --datapath ../data --dataset ${dataset} --base_model ${base_model} \
    --train_text few_shot_5 --train_ner few_shot_5 --few_shot_sets 10\
    --test_text test.words --test_ner test.ner \
    --test_example few_shot_5 --test_example_ner few_shot_5 \
    --epoch 10 --max_seq_len ${max_seq_len} --train_sup_cls_num 5 --test_sup_cls_num 5 \
    --sup_per_cls 2 --qur_per_cls 3 --episode_num 2  \
    --use_example True --just_eval False --load_model True \
    --lr 5e-05 --o_sent_ratio 0.0\
    --model_name save/prototype/5shot_${dataset}_naiveft_${base_model}_seq${max_seq_len}_epoch  \
    --load_dataset False \
    --load_model_name pretrained_models/prototype_pretrained_190.pt




