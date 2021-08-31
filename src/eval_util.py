import subprocess
from collections import defaultdict

# much of this adapted from https://github.com/iesl/dilated-cnn-ner/blob/08abe11aa8ecfd6eb499b959b5386f2d8ca0602e/src/eval_f1.py
def is_start(curr):
    return curr[0] == "B" or curr[0] == "U"

def is_continue(curr):
    return curr[0] == "I" or curr[0] == "L"

def is_background(curr):
    return not is_start(curr) and not is_continue(curr)

# def is_seg_start(curr, prev):
#     return ((is_start(curr) and not is_continue(curr))
#            or (is_continue(curr)
#                and (prev is None or is_background(prev) or prev[1:] != curr[1:])))

def is_seg_start(curr, prev):
    return (is_start(curr)
            or (is_continue(curr)
                and (prev is None or is_background(prev) or prev[1:] != curr[1:])))

def get_spans(sent):
    T = len(sent)
    spans = set()
    in_span, span_type = None, None
    for t in range(T):
        curr, prev = sent[t], sent[t-1] if t > 0 else None
        if is_seg_start(curr, prev):
            if in_span is not None:
                # tmp
                # if t-in_span != 1:
                spans.add((in_span, t, span_type))
            in_span, span_type = t, curr[2:]
        elif not is_continue(curr): # finished something
            if in_span is not None:
                #tmp
                # if t-in_span != 1:
                spans.add((in_span, t, span_type))
            in_span, span_type = None, None
    # catch stuff at the end
    if in_span is not None:
        #tmp
        # if t-in_span != 1:
        spans.add((in_span, T, span_type))
    return spans

# def batch_span_eval(pred, gold, idx2tag):
#     """
#     both preds and gold are T x bsz
#     """
#     T, bsz = pred.size()
#     assert gold.size(0) == T
#     assert gold.size(1) == bsz
#     npred, ngold, ncorrect = 0, 0, 0
#
#     for b in range(bsz):
#         predspans = get_spans([idx2tag[el.item()] for el in pred[:, b]])
#         goldspans = get_spans([idx2tag[el.item()] for el in pred[:, b]])
#         npred += len(predspans)
#         ngold += len(goldspans)
#         ncorrect += len(predspans & goldspans)
#
#     return npred, ngold, ncorrect


def batch_span_eval(pred, gold):
    """
    both preds and gold are bsz x T (but are lists)
    """
    bsz = len(pred)
    assert len(gold) == bsz
    # T = len(pred[0])
    npred, ngold, ncorrect = 0, 0, 0
    pred_span_per_type, gold_span_per_type, crct_span_per_type = defaultdict(int), defaultdict(int), defaultdict(int)
    for b in range(bsz):
        assert len(pred[b]) == len(gold[b])
        predspans = get_spans(pred[b])
        # if len(predspans) > 0:
        #     print(f'predspans: {predspans}')
        goldspans = get_spans(gold[b])
        # if len(goldspans) > 0:
        #     print(f'goldspans: {goldspans}')
        npred += len(predspans)
        ngold += len(goldspans)
        ncorrect += len(predspans & goldspans)
        # if len(predspans & goldspans) < len(predspans) or len(predspans & goldspans) < len(goldspans):
        #     print(pred[b])
        #     print(gold[b])
        for x in predspans:
            pred_span_per_type[x[2]] += 1
        for x in goldspans:
            gold_span_per_type[x[2]] += 1
        for x in predspans & goldspans:
            crct_span_per_type[x[2]] += 1
    return npred, ngold, ncorrect, pred_span_per_type, gold_span_per_type, crct_span_per_type


def batch_acc_eval(pred, gold):
    bsz = len(pred)
    assert len(gold) == bsz
    T = len(pred[0])
    crct = sum(pred[b][t] == gold[b][t] for b in range(bsz) for t in range(T))
    return T*bsz, crct


def run_conll(goldss, predss, outfi="tmppreds.txt"):
    with open(outfi, "w+") as f:
        for i, preds in enumerate(predss):
            golds = goldss[i]
            T = len(preds[0])
            for b in range(len(preds)):
                for t in range(T):
                    f.write("X O %s %s\n" % (golds[b][t], preds[b][t]))
                f.write("\n")
    outp = subprocess.check_output("perl conlleval.pl < %s" % outfi, shell=True)
    print(outp)
    outp = outp.decode('utf-8').split('\n')
    mainline = outp[1]
    scores = [float(thing.split(':')[1].strip(' %')) for thing in mainline.split(';')]
    acc, prec, rec, f1 = scores
    return acc, prec, rec, f1
