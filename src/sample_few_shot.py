import argparse
import os
import numpy as np

def get_label_dict(texts):
    label2id = {}
    id2label = {}
    orig_label2id = {}
    id2orig_label = {}
    idx = 0
    for text in texts:
        for line in text:
            tmp = line.strip().split()
            for ner_tag in tmp:  
                # if len(ner_tag.split('.')) > 1:
                #     ner_tag = ner_tag.split('-')[0]+'-'+ner_tag.split('.')[-1]      
                if ner_tag not in label2id:
                    if ner_tag == 'O':
                        label2id[ner_tag] = 0
                        id2label[0] = ner_tag
                    else:
                        try:
                            root = ner_tag.split('-')[1]
                        except IndexError:
                            print(ner_tag)
                            print(line)
                        if root not in orig_label2id:
                            orig_label2id[root] = idx
                            idx += 1                      
                            label2id['B-'+root] = orig_label2id[root] * 2 + 1
                            id2label[orig_label2id[root] * 2 + 1] = 'B-'+root
                            label2id['I-'+root] = orig_label2id[root] * 2 + 2
                            id2label[orig_label2id[root] * 2 + 2] = 'I-'+root
                            # label2id['B-'+root] = orig_label2id[root] * 2 + 1
                            # id2label[orig_label2id[root] * 2 + 1] = 'I-'+root
                            # label2id['I-'+root] = orig_label2id[root] * 2 + 1
                            # id2label[orig_label2id[root] * 2 + 2] = 'I-'+root

    print(id2label)
    print(label2id)
    return label2id, id2label

def get_dict(text):
    ner_dict = {}
    for i,line in enumerate(text):
        tmp = line.strip().split()
        for w in tmp:
            w = w.split('-')
            if len(w) > 2:
                print(w)
            if len(w) == 2:
                ner_tag = w[1]#.split('.')[-1]
                if ner_tag not in ner_dict:
                    ner_dict[ner_tag] = set()
                ner_dict[ner_tag].add(i)
    for ner_tag in ner_dict:
        ner_dict[ner_tag] = list(ner_dict[ner_tag])
    print([(k, len(ner_dict[k])) for k in ner_dict])
    print(len(ner_dict))
    return ner_dict

parser = argparse.ArgumentParser()
parser.add_argument("-data_path", default="../data/conll2003")
parser.add_argument("-data_file", default="train.words")
parser.add_argument("-ner_file", default="train.ner")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    few_shot_num = [10,20]
    with open(os.path.join(args.data_path, args.data_file)) as f:
        with open(os.path.join(args.data_path, args.ner_file)) as f_ner:
            text = f.readlines()
            ner_tags = f_ner.readlines()        
    ner_tag_dict = get_dict(ner_tags)
    
    for group in range(10):
        for few_shot in few_shot_num:
            out_file = 'few_shot_'+str(few_shot)+"_"+str(group)+".words"
            ner_file = 'few_shot_'+str(few_shot)+"_"+str(group)+".ner"
            with open(os.path.join(args.data_path, out_file), 'w') as fout1:
                with open(os.path.join(args.data_path, ner_file), 'w') as fout2:
                    for ner_tag in ner_tag_dict:
                        for i in range(few_shot):
                            idx = np.random.randint(len(ner_tag_dict[ner_tag]))
                            # while 'I-'+ner_tag not in ner_tags[ner_tag_dict[ner_tag][idx]]:
                            #     idx = np.random.randint(len(ner_tag_dict[ner_tag]))
                            fout1.write(text[ner_tag_dict[ner_tag][idx]])
                            fout2.write(ner_tags[ner_tag_dict[ner_tag][idx]])



        