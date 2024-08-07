import argparse
import json
import logging
import os
import random
import collections
import math
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import TestDataset
from model import KGReasoning, update_matrix, load_kbc
import time
import pickle
from collections import defaultdict
from tqdm import tqdm
from util import flatten_query, list2tuple, parse_time, set_global_seed, eval_tuple
from torchmetrics import SpearmanCorrCoef
import datetime

query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('r', 'r')): '2p',
                   ('e', ('r', 'r', 'r')): '3p',
                   ('e', ('r', 'r', 'r', 'r')): '4p',
                   ('e', ('r', 'r', 'r', 'r', 'r')): '5p',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                   (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                   (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM',
                   }
name_answer_dict = {'1p': ['e', ['r', ], 'e'],
                    '2p': ['e', ['r', 'e', 'r'], 'e'],
                    '3p': ['e', ['r', 'e', 'r', 'e', 'r'], 'e'],
                    '2i': [['e', ['r', ], 'e'], ['e', ['r', ], 'e'], 'e'],
                    '3i': [['e', ['r', ], 'e'], ['e', ['r', ], 'e'], ['e', ['r', ], 'e'], 'e'],
                    'ip': [[['e', ['r', ], 'e'], ['e', ['r', ], 'e'], 'e'], ['r', ], 'e'],
                    'pi': [['e', ['r', 'e', 'r'], 'e'], ['e', ['r', ], 'e'], 'e'],
                    '2in': [['e', ['r', ], 'e'], ['e', ['r', 'n'], 'e'], 'e'],
                    '3in': [['e', ['r', ], 'e'], ['e', ['r', ], 'e'], ['e', ['r', 'n'], 'e'], 'e'],
                    'inp': [[['e', ['r', ], 'e'], ['e', ['r', 'n'], 'e'], 'e'], ['r', ], 'e'],
                    'pin': [['e', ['r', 'e', 'r'], 'e'], ['e', ['r', 'n'], 'e'], 'e'],
                    'pni': [['e', ['r', 'e', 'r', 'n'], 'e'], ['e', ['r', ], 'e'], 'e'],
                    '2u-DNF': [['e', ['r', ], 'e'], ['e', ['r', ], 'e'], ['u', ], 'e'],
                    'up-DNF': [[['e', ['r', ], 'e'], ['e', ['r', ], 'e'], ['u', ], 'e'], ['r', ], 'e'],
                    }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(
    name_query_dict.keys())  # ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']
espace = 9
rspace = 11
mapping = dict()


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")
    parser.add_argument('--do_cp', action='store_true', help="do cardinality prediction")
    parser.add_argument('--path', action='store_true', help="do interpretation study")
    parser.add_argument('--cuda', action='store_true', help="use cuda")

    parser.add_argument('--train', action='store_true', help="do test")
    parser.add_argument('--data_path', type=str, default=None, help="KG data path")
    parser.add_argument('--dataset', type=str, default='icews18', help="dataset name")
    parser.add_argument('--kbc_path', type=str, default=None, help="kbc model path")
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int, help="used to speed up torch.dataloader")

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--fraction', type=int, default=1, help='fraction the entity to save gpu memory usage')
    parser.add_argument('--thrshd', type=float, default=0.001, help='thrshd for neural adjacency matrix')
    parser.add_argument('--neg_scale', type=int, default=1, help='scaling neural adjacency matrix for negation')
    parser.add_argument('--mask', type=str, default='mask', help='Using mask KGE score')
    parser.add_argument('--name', type=str, default='', help='name')
    parser.add_argument('--temp', type=int, default=15, help='name')

    parser.add_argument('--tasks', default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', type=str,
                        help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=12345, type=int, help="random seed")
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'],
                        help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')

    return parser.parse_args(args)


def log_metrics(mode, metrics, writer):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s: %f' % (mode, metric, metrics[metric]))
        print('%s %s: %f' % (mode, metric, metrics[metric]))
        writer.write('%s %s: %f\n' % (mode, metric, metrics[metric]))


def read_triples(filenames, nrelation, datapath):
    e2i = json.load(open(os.path.join(datapath, 'entity2id.json'), 'r'))
    r2i = json.load(open(os.path.join(datapath, 'relation2id.json'), 'r'))
    adj_list = [[] for i in range(nrelation)]
    adj_list_freq = [{} for i in range(nrelation)]
    edges_all = set()
    edges_vt = set()
    for filename in filenames:
        with open(filename) as f:
            for line in f.readlines():
                h, r, t, _ = line.strip().split('\t')
                #h, r, t, _ = line.strip().split()[:-1]
                h, r, t = e2i[h], r2i[r], e2i[t]
                adj_list[int(r)].append((int(h), int(t)))
                adj_list[int(r + nrelation / 2)].append((int(t), int(h)))
                if (h, t) in adj_list_freq[int(r)]:
                    adj_list_freq[int(r)][(h, t)] += 1
                    adj_list_freq[int(r + nrelation / 2)][(t, h)] += 1
                else:
                    adj_list_freq[int(r)][(h, t)] = 1
                    adj_list_freq[int(r + nrelation / 2)][(t, h)] = 1
    adj_list_freq_train = adj_list_freq.copy()
    for filename in ['valid.txt', 'test.txt']:
        with open(os.path.join(datapath, filename)) as f:
            for line in f.readlines():
                h, r, t, _ = line.strip().split('\t')
                #h, r, t, _ = line.strip().split()[:-1]
                h, r, t = e2i[h], r2i[r], e2i[t]
                edges_all.add((int(h), int(r), int(t)))
                edges_all.add((int(t), int(r + nrelation / 2), int(h)))
                edges_vt.add((int(t), int(r + nrelation / 2), int(h)))
                edges_vt.add((int(h), int(r), int(t)))
    with open(os.path.join(datapath, "train.txt")) as f:
        for line in f.readlines():
            h, r, t, _ = line.strip().split('\t')
            #h, r, t, _ = line.strip().split()[:-1]
            h, r, t = e2i[h], r2i[r], e2i[t]
            edges_all.add((int(h), int(r), int(t)))
            edges_all.add((int(t), int(r + nrelation / 2), int(h)))

    with open(os.path.join(datapath, "valid.txt")) as f:
        for line in f.readlines():
            h, r, t, _ = line.strip().split('\t')
            #h, r, t, _ = line.strip().split()[:-1]
            h, r, t = e2i[h], r2i[r], e2i[t]
            if (h, t) in adj_list_freq[int(r)]:
                adj_list_freq[int(r)][(h, t)] += 1
                adj_list_freq[int(r + nrelation / 2)][(t, h)] += 1
            else:
                adj_list_freq[int(r)][(h, t)] = 1
                adj_list_freq[int(r + nrelation / 2)][(t, h)] = 1

    return adj_list, edges_all, edges_vt, adj_list_freq_train, adj_list_freq


def count_freq(nrelation, datapath, time, name=None):
    e2i = json.load(open(os.path.join(datapath, 'entity2id.json'), 'r'))
    r2i = json.load(open(os.path.join(datapath, 'relation2id.json'), 'r'))
    ts2i = json.load(open(os.path.join(datapath, 'ts2id.json'), 'r'))
    adj_list_freq = [{} for i in range(nrelation)]
    filenames = ["train.txt"]
    #filenames = ["train.txt", "valid.txt"]
    #filenames = ["train.txt", "valid.txt", "test.txt"]
    for filename in filenames:
        with open(os.path.join(datapath, filename)) as f:
            for line in f.readlines():
                h, r, t, ts = line.strip().split('\t')
                #h, r, t, ts = line.strip().split()[:-1]
                h, r, t, ts = e2i[h], r2i[r], e2i[t], ts2i[ts]
                if ts == time:
                    break
                #freq = 1 / math.exp(ts - time + 1)
                freq = 1
                if (h, t) in adj_list_freq[int(r)]:
                    #freq = 1 / (time - ts)
                    adj_list_freq[int(r)][(h, t)] += freq
                    adj_list_freq[int(r + nrelation / 2)][(t, h)] += freq
                else:
                    adj_list_freq[int(r)][(h, t)] = 1
                    adj_list_freq[int(r + nrelation / 2)][(t, h)] = 1

    return adj_list_freq




def verify_chain(chain, chain_structure, edges_y, edges_p):  # (e, r, e, ..., e)
    '''
    verify the validity of the reasoning path (chain)
    '''
    global mapping
    head = chain[0]
    rel = 0
    neg = False
    judge = True
    edge_class = []
    for ele, ans_ele in zip(chain_structure[1:], chain[1:]):
        if ele == 'e':
            if neg:
                edge_judge = ((head, rel, ans_ele) not in edges_y)
                judge = judge & edge_judge
                if edge_judge:  # not in train/val/test
                    edge_class.append('y')
                elif (head, rel, ans_ele) in edges_p:  # in val/test
                    edge_class.append('p')
                else:  # in train
                    edge_class.append('n')
                neg = False
            else:
                edge_judge = ((head, rel, ans_ele) in edges_y)
                if edge_judge:
                    if (head, rel, ans_ele) in edges_p:  # in val/test
                        edge_class.append('p')
                    else:  # in train
                        edge_class.append('y')
                else:  # not in train/val/test
                    edge_class.append('n')
                judge = judge & edge_judge
            head = ans_ele
        elif ele == 'r':
            rel = ans_ele
        elif ele == 'n':
            neg = True

    chain_structure = chain_structure[1:-1]
    chain = chain[1:-1]
    out = ''
    neg = False
    edge_class = edge_class[::-1]
    idx = 0
    for ele, ans_ele in zip(chain_structure[::-1], chain[::-1]):
        if ele == 'e':
            out += '{:<9}'.format(str(ans_ele))
            mapping[str(ans_ele)] = id2ent[ans_ele]
        elif ele == 'r':
            if neg:
                out += '{:<11}'.format(edge_class[idx] + '<-r' + str(ans_ele) + '-X')
                neg = False
            else:
                out += '{:<11}'.format(edge_class[idx] + '<-r' + str(ans_ele) + '-')
            mapping['r' + str(ans_ele)] = id2rel[ans_ele]
            idx += 1
        elif ele == 'n':
            neg = True
    return judge, out


def verify(ans_structure, ans, edges_y, edges_p, offset=0):
    '''
    verify the validity of the reasoning path
    '''
    global mapping
    if ans_structure[1][0] == 'r':  # [[...], ['r', ...], 'e']
        chain_stucture = ['e'] + ans_structure[1] + ['e']
        if ans_structure[0] == 'e':  # ['e', ['r', ...], 'e']
            chain = [ans[0]] + ans[1] + [ans[2]]
            judge, out = verify_chain(chain, chain_stucture, edges_y, edges_p)
            out = '{:<9}'.format(str(ans[2])) + out + '{:<9}'.format(str(ans[0]))
            mapping[str(ans[2])] = id2ent[ans[2]]
            mapping[str(ans[0])] = id2ent[ans[0]]
            return judge, out
        else:
            chain = [ans[0][-1]] + ans[1] + [ans[2]]
            judge1, out1 = verify_chain(chain, chain_stucture, edges_y, edges_p)
            for ele in ans_structure[1] + [ans_structure[2]]:
                if ele == 'r':
                    offset += 11
                elif ele == 'e':
                    offset += 9
            judge2, out2 = verify(ans_structure[0], ans[0], edges_y, edges_p, offset)
            judge = judge1 & judge2
            out = '{:<9}'.format(str(ans[2])) + out1 + out2
            mapping[str(ans[2])] = id2ent[ans[2]]
            return judge, out

    else:  # [[...], [...], 'e']
        if ans_structure[-2][0] == 'u':
            union = True
            out = '{:<9}'.format(str(ans[-1]) + '(u)')
            ans_structure, ans = ans_structure[:-1], ans[:-1]
        else:
            union = False
            out = '{:<9}'.format(str(ans[-1]) + '(i)')
        mapping[str(ans[-1])] = id2ent[ans[-1]]
        judge = not union
        offset += 9
        for ele, ans_ele in zip(ans_structure[:-1], ans[:-1]):
            judge_ele, out_ele = verify(ele, ans_ele, edges_y, edges_p, offset)
            if union:
                judge = judge | judge_ele
            else:
                judge = judge & judge_ele
            out = out + out_ele + '\n' + ' ' * offset
        return judge, out


def get_cp_thrshd(model, tp_answers, fn_answers, args, dataloader, query_name_dict, device):
    '''
    get the best threshold for cardinality prediction on valid set
    '''
    probs = defaultdict(list)
    cards = defaultdict(list)
    best_thrshds = dict()
    for queries, queries_unflatten, query_structures in tqdm(dataloader):
        queries = torch.LongTensor(queries).to(device)
        embedding, _, _ = model.embed_query(queries, query_structures[0], 0)
        embedding = embedding.squeeze()
        hard_answer = tp_answers[queries_unflatten[0]]
        easy_answer = fn_answers[queries_unflatten[0]]
        num_hard = len(hard_answer)
        num_easy = len(easy_answer)

        probs[query_structures[0]].append(embedding.to('cpu'))
        cards[query_structures[0]].append(torch.tensor([num_hard + num_easy]))
    for query_structure in probs:
        prob = torch.stack(probs[query_structure])  # .to(device)
        card = torch.stack(cards[query_structure]).squeeze().to(torch.float)  # .to(device)
        ape = torch.zeros_like(card).to(torch.float).to(device)
        best_thrshd = 0
        best_mape = 10000
        nquery = prob.size(0)
        fraction = 10
        dim = nquery // fraction
        rest = nquery - fraction * dim
        for i in tqdm(range(10)):
            thrshd = i / 10
            for j in range(fraction):
                s = j * dim
                t = (j + 1) * dim
                if j == fraction - 1:
                    t += rest
                fractional_prob = prob[s:t, :].to(device)
                fractional_card = card[s:t].to(device)
                pre_card = (fractional_prob >= thrshd).to(torch.float).sum(-1)
                ape[s:t] = torch.abs(fractional_card - pre_card) / fractional_card
            mape = ape.mean()
            if mape < best_mape:
                best_mape = mape
                best_thrshd = thrshd
        best_thrshds[query_structure] = best_thrshd
    print(best_thrshds)
    return best_thrshds


def read(input, rel, ent):
    res = []
    for i in input:
        temp = []
        for j in i:
            if j == i[0]:
                temp.append(ent[j])
            else:
                temp.append(rel[j])
        res.append(temp)
    return res

def probability(cur_ts, nrelation, datapath, pes = None):
    e2i = json.load(open(os.path.join(datapath, 'entity2id.json'), 'r'))
    r2i = json.load(open(os.path.join(datapath, 'relation2id.json'), 'r'))
    ts2i = json.load(open(os.path.join(datapath, 'ts2id.json'), 'r'))
    facts = [{} for i in range(nrelation)]
    nums = [{} for i in range(nrelation)]
    '''cur_embedding = pes[cur_ts, :]
    inner_product = torch.matmul(cur_embedding, pes)
    inner_product = inner_product[: cur_ts]
    upper = torch.max(inner_product)'''
    ts_dict = datapath+'/ts_' + str(cur_ts) + '.pkl'
    ns_dict = datapath+'/ns_' + str(cur_ts) + '.pkl'
    filenames = ["train.txt", "valid.txt", "test.txt"]
    tau = 5
    for filename in filenames:
        with open(os.path.join(datapath, filename)) as f:
            for line in f.readlines():
                h, r, t, ts = line.strip().split('\t')
                #h, r, t, ts = line.strip().split()[:-1]
                h, r, t, ts = e2i[h], r2i[r], e2i[t], ts2i[ts]
                if ts >= cur_ts:
                    break
                # freq = 1 / math.exp(ts - time + 1)
                # freq = 1
                # prob = 1. / (cur_ts - ts)

                '''window = (cur_ts - ts) // 3
                prob = 1. / (window + 1)'''
                window = cur_ts - ts
                '''print('ts: ', ts)
                print('window: ', window)'''
                #prob = 1. / math.pow(window, 1/3)
                # prob = 1. / math.sqrt((window//10) + 1)
                #prob = 1. / math.exp(math.sqrt(window//tau))
                prob = 1. / math.sqrt(window)
                #prob = 1. / window
                #prob = 1. / math.exp(window)
                # freq = 1 / (time - ts)
                facts[int(r)][(h, t)] = prob
                facts[int(r + nrelation / 2)][(t, h)] = prob

                if (h, t) not in nums[int(r)]:
                    nums[int(r)][(h, t)] = 1
                    nums[int(r + nrelation / 2)][(t, h)] = 1
                else:
                    nums[int(r)][(h, t)] += 1
                    nums[int(r + nrelation / 2)][(t, h)] += 1

        '''with open(ts_dict, 'wb') as f:
            pickle.dump(facts, f)
        with open(ns_dict, 'wb') as f:
            pickle.dump(nums, f)'''
    #filenames = ["train.txt"]
    # filenames = ["train.txt", "valid.txt"]
    return facts, nums

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def evaluate(model, answers, args, dataloader, query_name_dict, device, writer, edges_y, edges_p,
             cp_thrshd):
    '''
    Evaluate queries in dataloader
    '''
    global mapping
    e2i = json.load(open(os.path.join(args.data_path, 'entity2id.json'), 'r'))
    r2i = json.load(open(os.path.join(args.data_path, 'relation2id.json'), 'r'))
    ts2i = json.load(open(os.path.join(args.data_path, 'ts2id.json'), 'r'))
    id2ent = {}
    id2rel = {}
    for k in e2i:
        id2ent[e2i[k]] = k
        id2ent[e2i[k] + len(e2i)] = k + '-1'

    for k in r2i:
        id2rel[r2i[k]] = k
        id2rel[r2i[k] + len(r2i)] = k + '-1'

    step = 0
    total_steps = len(dataloader)
    total_time = 0
    mrr = 0.0
    hits1 = 0.0
    hits3 = 0.0
    hits10 = 0.0
    sample_num = 0
    last_ts = -1
    '''pes = positionalencoding1d(d_model=128, length=len(ts2i))
    pes = pes.to(device)'''
    lambda_ = 1.0
    temperature = args.temp
    num_cands = 0
    num_rules = 0
    num_query = 0
    #global_embedding = [{} for i in range(args.nrelation)]
    #facts = collections.defaultdict(list)
    for queries, fired_rules, query_structures, query_pointer, filter_bias, query_heads, query_confs, query_constraints in tqdm(dataloader):
        '''print('Query: ', queries)
        print('Readable : ', id2ent[queries[0][0]], id2rel[queries[0][1]], id2ent[queries[0][2]])
        #print('fired rules: ', fired_rules)
        print('read rules: ', read(fired_rules, id2rel, id2ent))
        print('query structure: ', query_structures)
        print('COnfs: ', query_confs)'''
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        batch_heads_dict = collections.defaultdict(list)
        batch_confs_dict = collections.defaultdict(list)
        batch_constraints_dict = collections.defaultdict(list)

        num_query+=1

        cur_ts = queries[0][3]
        #cur_ts = 2975
        count = 0
        #changed = set()
        #print('cur_ts: ', cur_ts)
        if last_ts != cur_ts:
            #print('updating test ts: ', last_ts)

            facts, nums = probability(cur_ts, args.nrelation, args.data_path)

            dim = args.nentity // args.fraction

            for i in tqdm(range(len(facts))):
                rel_i = model.relation_embeddings[i]
                for (h, t) in facts[i]:
                    # count += 1
                    idx = h // dim
                    if idx >= args.fraction:
                        idx = args.fraction - 1
                    temp = rel_i[idx].to_dense()
                    idx_2 = h % dim
                    if idx == args.fraction - 1 and h >= args.fraction * dim:
                        idx_2 += dim

                    if (h,t) in nums[i]:
                        freq = nums[i][(h, t)]
                    else:
                        freq = 1
                    # if (h, t) not in global_embedding[i]:
                    #global_embedding[i][(h, t)] = temp[idx_2, t].to('cpu')
                    gl = temp[idx_2, t].to('cpu')
                    #gl = temp[idx_2, t]
                    #rel_i[idx][idx_2, t] = 5
                    #print(rel_i[idx].coalesce())
                    # global_embedding[i][(h, t)] = temp[idx_2, t]
                    # temp[idx_2, t] = facts[i][(h,t)] * lambda_ * math.sqrt(freq) + temp[idx_2, t] * (1 - lambda_)
                    lambda_freq = sigmoid(math.sqrt(freq - 1) / temperature) - 0.5
                    # lambda_freq = math.tanh(freq/temperature)
                    # temp[idx_2, t] = facts[i][(h, t)] * lambda_ + global_embedding[i][(h, t)] * (1 - lambda_) * freq
                    # temp[idx_2, t] = facts[i][(h, t)] * (1 - lambda_freq) + global_embedding[i][(h, t)] * lambda_freq * math.sqrt(freq)
                    #temp[idx_2, t] = facts[i][(h, t)] * lambda_ + global_embedding[i][(h, t)] * lambda_freq * math.sqrt((freq - 1))
                    temp[idx_2, t] = facts[i][(h, t)] * lambda_ + gl * lambda_freq * math.sqrt((freq - 1))
                    # temp[idx_2, t] = facts[i][(h, t)] * lambda_ + global_embedding[i][(h, t)] * lambda_freq
                    # temp[idx_2, t] = facts[i][(h, t)] * lambda_ / freq
                    #temp[idx_2, t] = facts[i][(h, t)]
                    temp[idx_2, t] = 1 if temp[idx_2, t] > 1 else temp[idx_2, t]
                    #temp[idx_2, t] = 1.

                    rel_i[idx] = temp.to_sparse().to('cuda')
                model.relation_embeddings[i] = rel_i
                # print('facts: ', facts)

        #facts[queries[0][1]].append((queries[0][0], queries[0][2]))
        #facts[queries[0][1] + args.nrelation // 2].append((queries[0][2], queries[0][0]))
        last_ts = cur_ts
        start = time.time()
        '''cur_ts = queries[0][3]
        if last_ts != -1 and last_ts != cur_ts:
            new_freq = count_freq(args.nrelation, args.data_path, cur_ts)
            model.relation_embeddings = []
            kbc_model = load_kbc(args.kbc_path, device, args.nentity, args.nrelation)
            for i in tqdm(range(args.nrelation)):
                relation_embedding = update_matrix(kbc_model, i, args.nentity, device, args.thrshd, new_freq[i])
                fractional_relation_embedding = []
                dim = args.nentity // args.fraction
                rest = args.nentity - args.fraction * dim
                for i in range(args.fraction):
                    s = i * dim
                    t = (i + 1) * dim
                    if i == args.fraction - 1:
                        t += rest
                    fractional_relation_embedding.append(relation_embedding[s:t, :].to_sparse().to(device))
                model.relation_embeddings.append(fractional_relation_embedding)
        last_ts = cur_ts'''

        # print('qsf: ', query_structures_flatten)
        for i, query in enumerate(fired_rules):
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
            batch_heads_dict[query_structures[i]].append(query_heads[i])
            batch_confs_dict[query_structures[i]].append(query_confs)
            batch_constraints_dict[query_structures[i]].append(query_constraints[i])
        # print("bqd: ", batch_queries_dict)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(
                    batch_queries_dict[query_structure]).cuda()
                batch_heads_dict[query_structure] = torch.LongTensor(
                    batch_heads_dict[query_structure]).cuda()

                #batch_constraints_dict[query_structure] = torch.LongTensor(batch_constraints_dict[query_structure]).cuda()

                batch_confs_dict[query_structure] = torch.FloatTensor(
                    batch_confs_dict[query_structure]).cuda()
                filter_bias = filter_bias.cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                batch_heads_dict[query_structure] = torch.LongTensor(batch_heads_dict[query_structure])
                batch_confs_dict[query_structure] = torch.FloatTensor(batch_confs_dict[query_structure])
                #batch_constraints_dict[query_structure] = torch.LongTensor(batch_constraints_dict[query_structure])

        all_idxs, all_embeddings = [], []
        for query_structure in batch_queries_dict:
            #queries = torch.LongTensor(batch_queries_dict[query_structure]).to(device)
            queries_temp = batch_queries_dict[query_structure]
            var_constraints = batch_constraints_dict[query_structure]
            #embedding, _, exec_query = model.embed_query(queries_temp, query_structure, 0, var_constraints)
            embedding, _, exec_query = model.embed_constrained_query(queries_temp, query_structure, 0, var_constraints)

            all_idxs.extend(batch_idxs_dict[query_structure])
            all_embeddings.append(embedding)

        '''qs = ('e', ('r',))
        q = [torch.LongTensor(queries)[ : 2].cuda().squeeze()]
        head_embedding, _, exec_query_head = model.embed_query(q, qs, 0)
        head_embedding = head_embedding.squeeze(0)'''

        embedding = torch.cat(all_embeddings, dim=0)
        #print('all idx: ', all_idxs)
        rearrange = torch.argsort(torch.LongTensor(all_idxs))
        #print('all idx check: ', torch.LongTensor(all_idxs)[rearrange])
        embedding = embedding[rearrange].squeeze(1)
        #print('Embeddings: ', embedding)
        #v_, i_ = torch.sort(embedding, descending=True, dim=1)
        #print('Embeddings Descending: ', v_[:, :10])
        #print('IDX: ', i_[:, :10])

        '''conf = torch.relu(head_embedding - embedding)
        conf = torch.sum(conf, dim=1, keepdim=True)
        conf = 1 - torch.tanh(conf)'''

        conf = torch.FloatTensor(query_confs).cuda().unsqueeze(-1)
        #print(embedding)
        #print('confs: ', conf)
        embedding = embedding * conf
        end = time.time()
        total_time = total_time + end - start
        num_cands += torch.sum(embedding != 0)
        num_rules += embedding.shape[0]
        #print(num_rules)

        final_logit = []
        pointer = 0
        for i in range(len(query_pointer)):
            temp_logit = embedding[pointer: pointer + query_pointer[i], :]
            #temp_logit = torch.softmax(temp_logit, dim=-1)
            #temp_logit = torch.mean(temp_logit, dim=0, keepdim=True)
            #print(temp_logit.shape)
            #print('temp_0: ', temp_logit)
            temp_logit = torch.ones_like(temp_logit).cuda() - temp_logit
            '''print('before: ', temp_logit.shape)
            print('temp: ', temp_logit)
            print('prod: ', torch.prod(temp_logit, dim=0, keepdim=True))'''
            temp_logit = 1 - torch.prod(temp_logit, dim=0, keepdim=True)
            #print('after: ', temp_logit.shape)
            #temp_logit = temp_logit.squeeze(1)

            final_logit.append(temp_logit)
            pointer += query_pointer[i]

        final_logit = torch.cat(final_logit, dim=0).cuda()
        final_logit += filter_bias.cuda()
        # temp_mrr, temp_hits1, temp_hits3, temp_hits10 = calc_filtered_mrr(num_entity=args.nentity, score=final_logit, queries=np.array(queries), quads_to_filter=quads_to_filter)
        answers_ = torch.LongTensor(queries)
        answers_test = answers_[:, 2]
        target_score = final_logit[0, answers_test]
        #print(final_logit)
        #argsort = torch.argsort(final_logit, dim=1, descending=True)
        sort_score = torch.sort(final_logit, dim=1 ,descending=True)[0]
        query_num = len(queries)

        #answers = queries
        '''if args.cuda:
            answers = answers.cuda()'''
        #answers = answers[:, 2]
        for i in range(query_num):
            #ranking = (argsort[i, :] == answers_test[i]).nonzero()
            #print('target: ', target_score)
            #print((sort_score[i, :] == target_score[i]).nonzero())
            if target_score[i] == 0.0:
                temp_mrr = 0.0
                temp_hits1 = 0.0
                temp_hits3 = 0.0
                temp_hits10 = 0.0
            else:
                ranking = (sort_score[i, :] == target_score[i]).nonzero()[0]
                ranking = 1 + ranking.item()
                temp_mrr = 1.0 / ranking
                temp_hits1 = 1.0 if ranking <= 1 else 0.0
                temp_hits3 = 1.0 if ranking <= 3 else 0.0
                temp_hits10 = 1.0 if ranking <= 10 else 0.0
            mrr += temp_mrr
            hits1 += temp_hits1
            hits3 += temp_hits3
            hits10 += temp_hits10

        sample_num += query_num

        if step % 1000 == 0:
            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

        step += 1


    metrics = {}
    metrics['MRR'] = mrr / sample_num
    metrics['Hits1'] = hits1 / sample_num
    metrics['Hits3'] = hits3 / sample_num
    metrics['Hits10'] = hits10 / sample_num
    writer.write(str(metrics))
    print('total time: ', total_time)
    print('per cands: ', num_cands / num_query)
    print('per rules: ', num_rules / num_query)
    return metrics

def load_quadruples(inPath, fileName, nrelation, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        entity2id = json.load(open(os.path.join(inPath, 'entity2id.json'), 'r'))
        relation2id = json.load(open(os.path.join(inPath, 'relation2id.json'), 'r'))
        ts2id = json.load(open(os.path.join(inPath, 'ts2id.json'), 'r'))
        for line in fr:
            line_split = line.strip().split('\t')
            #line_split = line.strip().split()
            head = int(entity2id[line_split[0]])
            tail = int(entity2id[line_split[2]])
            rel = int(relation2id[line_split[1]])
            time = int(ts2id[line_split[3]])
            quadrupleList.append((head, rel, tail, time))
            quadrupleList.append((tail, rel + nrelation, head, time))
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.strip().split('\t')
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append((head, rel, tail, time))
                times.add(time)
    times = list(times)
    times.sort()

    return quadrupleList, np.asarray(times)


def load_data(args):
    '''
    Load queries and remove queries not in tasks
    '''
    logging.info("loading data")
    # train_queries_1 = pickle.load(open(os.path.join(args.data_path, "train_noflatten_queries_.pkl"), 'rb'))
    # train_ori = [i[1] for i in train_queries_1]
    # train_queries = np.array(train_queries_1, dtype=object)[:, 0]
    valid_queries = pickle.load(open(os.path.join(args.data_path, "valid_repro_123_inverse_tr_0.01queries_conf_constraints_0.pkl"), 'rb'))
    valid_answers = pickle.load(open(os.path.join(args.data_path, "valid_repro_123_inverse_tr_0.01answers_conf_constraints_0.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(args.data_path, "test_repro_123_inverse_tr_0.01queries_conf_constraints_0.pkl"), 'rb'))
    test_answers = pickle.load(open(os.path.join(args.data_path, "test_repro_123_inverse_tr_0.01answers_conf_constraints_0.pkl"), 'rb'))
    valid_queries = sorted(valid_queries, key=lambda x : x[0][3])
    test_queries = sorted(test_queries, key=lambda x : x[0][3])

    # remove tasks not in args.task
    return valid_queries, valid_answers, test_queries, test_answers


def main(args):
    set_global_seed(args.seed)
    tasks = args.tasks.split('.')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    dataset_name = args.dataset
    filename = 'results/' + dataset_name + '_' + str(args.name) + '_'+ str(args.fraction) + '_' + str(args.thrshd) + '_' + str(datetime.datetime.now()) + '.txt'
    writer = open(filename, 'a+')

    with open('%s/stats.txt' % args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].strip().split('\t')[0])
        nrelation = int(entrel[0].strip().split('\t')[1])

    global id2ent, id2rel
    '''with open('%s/id2ent.pkl' % args.data_path, 'rb') as f:
        id2ent = pickle.load(f)
    with open('%s/ent2id.pkl' % args.data_path, 'rb') as f:
        ent2id = pickle.load(f)
    with open('%s/id2rel.pkl' % args.data_path, 'rb') as f:
        id2rel = pickle.load(f)'''

    args.nentity = nentity
    args.nrelation = nrelation * 2

    adj_list, edges_y, edges_p, freq_train, freq_valid = read_triples([os.path.join(args.data_path, "train.txt"), os.path.join(args.data_path, "valid.txt")], args.nrelation,
                                              args.data_path)

    #freq_train = count_freq(args.nrelation, args.data_path, 314)

    '''adj_list_v, edges_y_v, edges_p_v = read_triples([os.path.join(args.data_path, "valid.txt")], args.nrelation,
                                              args.data_path)'''


    valid_queries, valid_answers, test_queries, test_answers = load_data(args)
    valid_quads = load_quadruples('./data/{}'.format(args.dataset), 'valid.txt', nrelation)[0]
    test_quads = load_quadruples('./data/{}'.format(args.dataset), 'test.txt', nrelation)[0]

    valid_dataloader = DataLoader(
        TestDataset(
            valid_queries,
            args.nentity,
            args.nrelation,
            valid_quads
        ),
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num,
        collate_fn=TestDataset.collate_fn
    )

    test_dataloader = DataLoader(
        TestDataset(
            test_queries,
            args.nentity,
            args.nrelation,
            test_quads
        ),
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num,
        collate_fn=TestDataset.collate_fn
    )

    model = KGReasoning(args, device, adj_list, query_name_dict, name_answer_dict, freq_valid)

    '''dim = args.nentity // args.fraction
    for i in tqdm(range(args.nrelation)):
        freq_mat_i = freq_train[i]
        rel_i = model.relation_embeddings[i]
        for k in freq_mat_i:
            idx = k[0] // dim
            if idx >= args.fraction:
                idx = args.fraction - 1
            temp = rel_i[idx].to_dense()
            idx_2 = k[0] % dim
            if idx == args.fraction - 1 and k[0] >= args.fraction * dim:
                idx_2 += dim
            temp[idx_2, k[1]] *= freq_mat_i[k]
            rel_i[idx] = temp.to_sparse()
        softmax = nn.Softmax(dim=1)
        relation_embedding = softmax(relation_embedding)'''


    '''print('Adding Valid: ')
    count = 0
    count_change = 0
    dim = args.nentity // args.fraction
    for i in tqdm(range(args.nrelation)):
        rel_i = model.relation_embeddings[i]
        for (h, t) in adj_list_v[i]:
            count += 1
            idx = h // dim
            if idx >= args.fraction:
                idx = args.fraction - 1
            temp = rel_i[idx].to_dense()
            idx_2 = h % dim
            if idx == args.fraction - 1 and h >= args.fraction * dim:
                idx_2 += dim
            if temp[idx_2, t] == 1.:
                count_change += 1
            temp[idx_2, t] = 1.
            rel_i[idx] = temp.to_sparse()
        model.relation_embeddings[i] = rel_i

    print('count: ', count)
    print('count_change: ', count_change)'''

    '''print('Erasing: ')
    for i in tqdm(range(args.nrelation)):
        rel_i = model.relation_embeddings[i]
        for idx in range(len(rel_i)):
            temp = rel_i[idx].to_dense()
            mask = temp >= 1.
            temp = temp * mask
            rel_i[idx] = temp.to_sparse()
        model.relation_embeddings[i] = rel_i'''


    cp_thrshd = None

    metrics = evaluate(model, test_answers, args, test_dataloader, query_name_dict, device, writer,
             edges_y, edges_p, cp_thrshd)

    print(args.name)
    print(metrics)



if __name__ == '__main__':
    main(parse_args())