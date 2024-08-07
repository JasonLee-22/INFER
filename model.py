import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import pickle
import math
import collections
import itertools
import time
from tqdm import tqdm
import os
import sys
import json
sys.path.append('rp')
from kbc.src.models import ComplEx

def load_kbc(model_path, device, nentity, nrelation):
    model = ComplEx(sizes=(nentity, nrelation, nentity), rank=1000, init_size=1e-3)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

@torch.no_grad()
def kge_forward(model, h, r, device, nentity):
    bsz = h.size(0)
    r = r.unsqueeze(-1).repeat(bsz, 1)
    h = h.unsqueeze(-1)
    positive_sample = torch.cat((h, r, h), dim=1)
    score = model(positive_sample, score_rhs=True, score_rel=False, score_lhs=False)
    return score[0]

@torch.no_grad()
def neural_adj_matrix(model, rel, nentity, device, thrshd, adj_list, freq, mask_ = False):
    bsz = 1000
    softmax = nn.Softmax(dim=1)
    sigmoid = nn.Sigmoid()
    relation_embedding = torch.zeros(nentity, nentity).to(torch.float)
    #return relation_embedding
    r = torch.LongTensor([rel]).to(device)
    num = torch.zeros(nentity, 1).to(torch.float).to(device)
    adj_for_mask = [[] for x in range(nentity)]
    if mask_:
        for (h, t) in adj_list:
            adj_for_mask[h].append(t)
    else:
        for (h, t) in adj_list:
            num[h, 0] += 1

    num = torch.maximum(num, torch.ones(nentity, 1).to(torch.float).to(device))
    for s in range(0, nentity, bsz):
        t = min(nentity, s+bsz)
        h = torch.arange(s, t).to(device)
        score = kge_forward(model, h, r, device, nentity)
        if mask_[-4: ] == 'mask':
            #mask_norm = torch.ones_like(score).to(device)
            mask_norm = torch.zeros_like(score).to(device)
            for index, ents in enumerate(adj_for_mask[s : t]):
                for e in ents:
                    #mask_norm[index][e] = 0
                    mask_norm[index][e] = float("-inf")
            #score = (score * mask_norm).to(torch.float)
            score = (score + mask_norm).to(torch.float)
            normalized_score = softmax(score)
        else:
            #normalized_score = softmax(score) * num[s:t, :]
            #normalized_score = softmax(score)
            #normalized_score = softmax(score)
            normalized_score = softmax(score)

        mask = (normalized_score >= thrshd).to(torch.float)
        normalized_score = mask * normalized_score
        #normalized_score = softmax(normalized_score)
        relation_embedding[s:t, :] = normalized_score.to('cpu')
    #relation_embedding = softmax(relation_embedding)
    #print('neu: ', torch.sort(relation_embedding, descending=True)[0])
    return relation_embedding


@torch.no_grad()
def update_matrix(model, rel, nentity, device, thrshd, freq):
    bsz = 1000
    softmax = nn.Softmax(dim=1)
    relation_embedding = torch.zeros(nentity, nentity).to(torch.float)
    r = torch.LongTensor([rel]).to(device)

    for s in range(0, nentity, bsz):
        t = min(nentity, s+bsz)
        h = torch.arange(s, t).to(device)
        h_fast = set(range(s ,t))
        score = kge_forward(model, h, r, device, nentity)            #normalized_score = softmax(score) * num[s:t, :]
        #score = torch.zeros(t - s, t - s).to(torch.float)
        for k in freq:
            if k[0] in h_fast:
                score[k[0] - s][k[1]] *= freq[k]
        normalized_score = softmax(score)
        mask = (normalized_score >= thrshd).to(torch.float)
        normalized_score = mask * normalized_score

        #normalized_score = softmax(normalized_score)
        relation_embedding[s:t, :] = normalized_score.to('cpu')
    #relation_embedding = softmax(relation_embedding)
    return relation_embedding

@torch.no_grad()
def healthy_softmax(relation_embedding, freq):
    #norm_relation = torch.zeros_like(relation_embedding).to(torch.float).to('cpu')
    for k in freq:
        relation_embedding[k[0]][k[1]] *= freq[k]
    exp = torch.exp(relation_embedding)
    summed = torch.sum(relation_embedding, dim=1, keepdim=True)
    relation_embedding = exp / summed
    del exp
    '''norm_relation = relation_embedding.clone()
    norm_relation = norm_relation.numpy()
    softmax_output = np.exp(norm_relation) / np.sum(np.exp(norm_relation), axis=1, keepdims=True)'''
    '''softmax = nn.Softmax(dim=1)
    bsz = 10
    end = len(relation_embedding)
    for s in range(0, end, bsz):
        t = min(end, s + bsz)
        temp = relation_embedding[s:t, :].clone().to('cpu')
        temp = softmax(temp)
        norm_relation[s:t, :] = temp'''
    #torch.exp(relation_embedding, out=relation_embedding)
    '''relation_embedding = np.exp(relation_embedding.cpu().numpy())
    relation_embedding = torch.FloatTensor(relation_embedding).cuda()'''
    #summed = torch.sum(relation_embedding, dim=1, keepdim=True)
    #relation_embedding /= summed
    # softmax = nn.Softmax(dim=1)
    #relation_embedding = torch.softmax(relation_embedding, dim=-1)
    return relation_embedding


class KGReasoning(nn.Module):
    def __init__(self, args, device, adj_list, query_name_dict, name_answer_dict, freq):
        super(KGReasoning, self).__init__()
        self.nentity = args.nentity
        self.nrelation = args.nrelation
        self.device = device
        self.relation_embeddings = list()
        self.fraction = args.fraction
        self.query_name_dict = query_name_dict
        self.name_answer_dict = name_answer_dict
        self.neg_scale = args.neg_scale
        self.freq_mat = freq
        self.ori_embeddings = []
        dataset_name = args.dataset
        filename = 'neural_adj/'+dataset_name+'_'+str(args.fraction)+'_'+str(args.thrshd) + '_'+ str(args.mask)+'.pt'
        if os.path.exists(filename):
            self.relation_embeddings = torch.load(filename)
        else:
            kbc_model = load_kbc(args.kbc_path, device, args.nentity, args.nrelation)
            #kbc_model = None
            for i in tqdm(range(args.nrelation)):
                relation_embedding = neural_adj_matrix(kbc_model, i, args.nentity, device, args.thrshd, adj_list[i], freq[i], mask_=args.mask)
                #self.ori_embeddings.append(relation_embedding)

                #relation_embedding = (relation_embedding>=1).to(torch.float) * 0.9999 + (relation_embedding<1).to(torch.float) * relation_embedding
                #relation_embedding = healthy_softmax(relation_embedding, self.freq_mat[i])
                '''for (h, t) in adj_list[i]:
                    relation_embedding[h, t] = 1.'''
                #add fractional
                fractional_relation_embedding = []
                dim = args.nentity // args.fraction
                rest = args.nentity - args.fraction * dim
                for i in range(args.fraction):
                    s = i * dim
                    t = (i+1) * dim
                    if i == args.fraction - 1:
                        t += rest
                    fractional_relation_embedding.append(relation_embedding[s:t, :].to_sparse().to(self.device))
                self.relation_embeddings.append(fractional_relation_embedding)
            torch.save(self.relation_embeddings, filename)

    def relation_projection(self, embedding, r_embedding, is_neg=False):
        dim = self.nentity // self.fraction
        rest = self.nentity - self.fraction * dim
        new_embedding = torch.zeros_like(embedding).to(self.device)
        r_argmax = torch.zeros(self.nentity).to(self.device)
        for i in range(self.fraction):
            s = i * dim
            t = (i+1) * dim
            if i == self.fraction - 1:
                t += rest
            fraction_embedding = embedding[:, s:t]
            if fraction_embedding.sum().item() == 0:
                continue
            nonzero = torch.nonzero(fraction_embedding, as_tuple=True)[1]
            fraction_embedding = fraction_embedding[:, nonzero]
            fraction_r_embedding = r_embedding[i].to(self.device).to_dense()[nonzero, :].unsqueeze(0)
            if is_neg:
                fraction_r_embedding = torch.minimum(torch.ones_like(fraction_r_embedding).to(torch.float), self.neg_scale*fraction_r_embedding)
                fraction_r_embedding = 1. - fraction_r_embedding
            fraction_embedding_premax = fraction_r_embedding * fraction_embedding.unsqueeze(-1)
            fraction_embedding, tmp_argmax = torch.max(fraction_embedding_premax, dim=1)
            tmp_argmax = nonzero[tmp_argmax.squeeze()] + s
            new_argmax = (fraction_embedding > new_embedding).to(torch.long).squeeze()
            r_argmax = new_argmax * tmp_argmax + (1-new_argmax) * r_argmax
            new_embedding = torch.maximum(new_embedding, fraction_embedding)
        return new_embedding, r_argmax.cpu().numpy()
    
    def intersection(self, embeddings):
        return torch.prod(embeddings, dim=0)

    def union(self, embeddings):
        return (1. - torch.prod(1.-embeddings, dim=0))

    def embed_constrained_query(self, queries, query_structure, idx, var_constraints):
        all_relation_flag = True
        exec_query = []
        decay_co = 0.9
        for ele in query_structure[
            -1]:  # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            embeddings_out = []
            for p in range(len(queries)):
                idx_in = idx
                temp_query = queries[p].unsqueeze(0).to(self.device)
                if query_structure[0] == 'e':
                    bsz = temp_query.size(0)
                    embedding = torch.zeros(bsz, self.nentity).to(torch.float).to(
                        self.device)  # starting point distribution
                    # embedding.scatter_(-1, queries[:, idx].unsqueeze(-1), 1)
                    embedding.scatter_(-1, temp_query[:, idx_in].unsqueeze(-1), 1)
                    # exec_query.append(queries[:, idx])
                    exec_query.append(temp_query[:, idx_in])
                    idx_in += 1

                r_exec_query = []
                if len(var_constraints[p]) == 0:
                    for i in range(len(query_structure[-1])):
                        if query_structure[-1][i] == 'n':
                            assert (queries[:, idx_in] == -2).all()
                            r_exec_query.append('n')
                        else:
                            # r_embedding = np.array(self.relation_embeddings)[queries[:, idx]]
                            # r_embedding = torch.index_select(self.relation_embeddings, dim=0, index=queries[:, idx])
                            r_embedding = self.relation_embeddings[temp_query[0, idx_in]]

                            if (i < len(query_structure[-1]) - 1) and query_structure[-1][i + 1] == 'n':
                                embedding, r_argmax = self.relation_projection(embedding, r_embedding, True)
                            else:
                                embedding, r_argmax = self.relation_projection(embedding, r_embedding, False)
                            #if len(query_structure[-1]) > 1:
                                '''print('lenght of the rule: ', len(query_structure[-1]))
                                print('query: ', temp_query)
                                print('current step: ', i, temp_query[0, idx_in])'''
                                #temp = torch.sort(embedding, dim=1 ,descending=True)
                                '''print('top 10: ')
                                print(temp[0][0][:5])
                                print(temp[1][0][:5])'''
                            r_exec_query.append((temp_query[0, idx_in].item(), r_argmax))
                            r_exec_query.append('e')

                            #embedding = embedding * decay_co

                        idx_in += 1
                    #print('no vc: ', embedding.unsqueeze(0).shape)
                    embeddings_out.append(embedding.unsqueeze(0))
                    r_exec_query.pop()
                    exec_query.append(r_exec_query)
                    exec_query.append('e')

                else:
                    temp_constraints = var_constraints[p]
                    if len(temp_constraints) == 1:
                        if temp_constraints[0] == [0,2]:
                            r_embedding = self.relation_embeddings[temp_query[0, idx_in]]
                            embedding, r_argmax = self.relation_projection(embedding, r_embedding, False)

                            #embedding = embedding * decay_co

                            r_exec_query.append((temp_query[0, idx_in].item(), r_argmax))
                            r_exec_query.append('e')
                            idx_in += 1

                            r_embedding = self.relation_embeddings[temp_query[0, idx_in]]
                            dim = self.nentity // self.fraction
                            rest = self.nentity - self.fraction * dim
                            for i in range(self.fraction):
                                s = i * dim
                                t = (i + 1) * dim
                                if i == self.fraction - 1:
                                    t += rest
                                fraction_embedding = embedding[:, s:t].squeeze()
                                fraction_r_embedding = r_embedding[i].to(self.device).to_dense()[:, temp_query[0, 0]]   # probability to the first entity (comlying to constraints)
                                embedding[:, s:t] = fraction_embedding * fraction_r_embedding
                            idx_in += 1

                            #embedding = embedding * decay_co

                            r_embedding = self.relation_embeddings[temp_query[0, idx_in]]

                            bsz = temp_query.size(0)        #max first entity
                            #print('bsz: ', bsz)
                            temp_embedding = torch.zeros(bsz, self.nentity).to(torch.float).to(self.device)
                            temp_embedding.scatter_(-1, temp_query[:, 0].unsqueeze(-1), torch.max(embedding).item())


                            embedding, r_argmax = self.relation_projection(temp_embedding, r_embedding, False)
                            r_exec_query.append((temp_query[0, idx_in].item(), r_argmax))
                            r_exec_query.append('e')

                            #embedding = embedding * decay_co

                            #print('vc [0,2]: ', embedding.unsqueeze(0).shape)
                            embeddings_out.append(embedding.unsqueeze(0))

                        elif temp_constraints[0] == [1, 2]:
                            new_embeddings = torch.zeros(bsz, self.nentity).to(torch.float).to(
                                self.device)
                            r_embedding = self.relation_embeddings[temp_query[0, idx_in]]
                            embedding, r_argmax = self.relation_projection(embedding, r_embedding, False)
                            # embedding = embedding * decay_co
                            r_exec_query.append((temp_query[0, idx_in].item(), r_argmax))
                            r_exec_query.append('e')
                            idx_in += 1

                            r_embedding = self.relation_embeddings[temp_query[0, idx_in]]
                            dim = self.nentity // self.fraction
                            rest = self.nentity - self.fraction * dim
                            for i in range(self.fraction):
                                s = i * dim
                                t = (i + 1) * dim
                                if i == self.fraction - 1:
                                    t += rest
                                fraction_embedding = embedding[:, s:t].squeeze()
                                fraction_r_embedding = r_embedding[i].to(self.device).to_dense()
                                for j in range(0, t-s):
                                    new_embeddings[0, s + j] = fraction_embedding[j] * fraction_r_embedding[j][s+j]
                                #embedding[:, s:t] = fraction_embedding * fraction_r_embedding
                            idx_in += 1

                            if len(query_structure[-1]) == 4:
                                r_embedding = self.relation_embeddings[temp_query[0, idx_in]]
                                new_embeddings, r_argmax = self.relation_projection(new_embeddings, r_embedding, False)
                                #new_embeddings = new_embeddings.unsqueeze(0)

                            embeddings_out.append(new_embeddings.unsqueeze(0))



                        elif temp_constraints[0] == [0, 1]:
                            r_embedding = self.relation_embeddings[temp_query[0, idx_in]]
                            embedding, r_argmax = self.relation_projection(embedding, r_embedding, False)
                            # embedding = embedding * decay_co
                            r_exec_query.append((temp_query[0, idx_in].item(), r_argmax))
                            r_exec_query.append('e')
                            new_embeddings = torch.zeros(bsz, self.nentity).to(torch.float).to(
                        self.device)
                            new_embeddings[0, temp_query[0, idx_in]] = embedding[0, temp_query[0, idx_in]]
                            idx_in += 1
                            for i in range(2, len(query_structure[-1])):
                                if query_structure[-1][i] == 'n':
                                    assert (queries[:, idx_in] == -2).all()
                                    r_exec_query.append('n')
                                else:
                                    r_embedding = self.relation_embeddings[temp_query[0, idx_in]]

                                    if (i < len(query_structure[-1]) - 1) and query_structure[-1][i + 1] == 'n':
                                        embedding, r_argmax = self.relation_projection(embedding, r_embedding, True)
                                    else:
                                        embedding, r_argmax = self.relation_projection(embedding, r_embedding, False)
                                    r_exec_query.append((temp_query[0, idx_in].item(), r_argmax))
                                    r_exec_query.append('e')
                                idx_in += 1
                            embeddings_out.append(embedding.unsqueeze(0))

                        elif temp_constraints[0] == [1, 3]:
                            result = []
                            r_embedding = self.relation_embeddings[temp_query[0, idx_in]]
                            next_r_embedding = self.relation_embeddings[temp_query[0, idx_in + 1]]
                            last_r_embedding = self.relation_embeddings[temp_query[0, idx_in + 2]]


                            embedding, r_argmax = self.relation_projection(embedding, r_embedding, False)

                            #embedding = embedding * decay_co

                            r_exec_query.append((temp_query[0, idx_in].item(), r_argmax))
                            r_exec_query.append('e')
                            idx_in += 1

                            dim = self.nentity // self.fraction
                            rest = self.nentity - self.fraction * dim

                            r3_embedding = []
                            for i in range(self.fraction):
                                s = i * dim
                                t = (i + 1) * dim
                                if i == self.fraction - 1:
                                    t += rest
                                r3_embedding.append(last_r_embedding[i].to(self.device).to_dense())
                            r3_embedding = torch.cat(r3_embedding, dim=0)

                            #r3_embedding = r3_embedding * decay_co

                            for i in range(self.fraction):
                                s = i * dim
                                t = (i + 1) * dim
                                if i == self.fraction - 1:
                                    t += rest
                                fraction_embedding = embedding[:, s:t].squeeze()
                                fraction_r_embedding = next_r_embedding[i].to(self.device).to_dense()

                                #fraction_r_embedding = fraction_r_embedding * decay_co

                                fraction_all_probabilty = fraction_embedding.unsqueeze(-1) * fraction_r_embedding

                                temp_r3 = r3_embedding[:, s:t]
                                final_all_prob = fraction_all_probabilty * temp_r3.transpose(1,0)
                                #print(final_all_prob.shape)
                                final_all_prob = torch.max(final_all_prob, dim=1)[0]
                                result.append(final_all_prob.unsqueeze(0))

                            temp = torch.cat(result, dim=1)
                            #print('vc [1,3]: ', temp.shape)
                            embeddings_out.append(temp.unsqueeze(0))
                            '''r3_embedding = []
                                                            for j in range(self.fraction):
                                                                fraction_next_r_embedding = next_r_embedding[j].to(self.device).to_dense()[:, s:t]  #(dim, nentity)
                                                                r3_embedding.append(fraction_next_r_embedding)
                                                            r3_embedding = torch.cat(r3_embedding, dim=0)    #(nentity, dim)

                                                            fraction_all_probabilty = fraction_all_probabilty * r3_embedding.transpose(1, 0)
                                                            embedding[:, s:t] = torch.max(fraction_all_probabilty, dim=-1)[0]'''

                            #embeddings_out.append(embedding.unsqueeze(0))
                    else:
                        r_embedding = self.relation_embeddings[temp_query[0, idx_in]]
                        embedding, r_argmax = self.relation_projection(embedding, r_embedding, False)
                        r_exec_query.append((temp_query[0, idx_in].item(), r_argmax))
                        r_exec_query.append('e')

                        #embedding = embedding * decay_co

                        idx_in += 1

                        r_embedding = self.relation_embeddings[temp_query[0, idx_in]]
                        dim = self.nentity // self.fraction
                        rest = self.nentity - self.fraction * dim
                        for i in range(self.fraction):
                            s = i * dim
                            t = (i + 1) * dim
                            if i == self.fraction - 1:
                                t += rest
                            fraction_embedding = embedding[:, s:t].squeeze()
                            fraction_r_embedding = r_embedding[i].to(self.device).to_dense()[:, temp_query[0, 0]]  # probability to the first entity (comlying to constraints)
                            embedding[:, s:t] = fraction_embedding * fraction_r_embedding
                        idx_in += 1

                        #embedding = embedding * decay_co

                        r_embedding = self.relation_embeddings[temp_query[0, idx_in]]
                        entity = temp_query[0, 0]
                        entity_idx = entity // dim
                        if entity_idx == self.fraction:
                            entity_idx -= 1
                        fraction_r_embedding = r_embedding[entity_idx].to(self.device).to_dense()
                        idx_2 = entity % dim
                        if entity_idx == self.fraction - 1 and entity >= self.fraction * dim:
                            idx_2 += dim
                        fraction_r_embedding = fraction_r_embedding[idx_2, :]
                        embedding = embedding * fraction_r_embedding

                        #embedding = embedding * decay_co

                        #print('vc full: ', embedding.unsqueeze(0).shape)
                        embeddings_out.append(embedding.unsqueeze(0))


        return torch.cat(embeddings_out, dim=0), idx, exec_query

    def embed_query(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        exec_query = []
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            embeddings_out = []
            for p in range(len(queries)):
                idx_in = idx
                temp_query = queries[p].unsqueeze(0).to(self.device)
                if query_structure[0] == 'e':
                    bsz = temp_query.size(0)
                    embedding = torch.zeros(bsz, self.nentity).to(torch.float).to(
                        self.device)  # starting point distribution
                    #embedding.scatter_(-1, queries[:, idx].unsqueeze(-1), 1)
                    embedding.scatter_(-1, temp_query[:, idx_in].unsqueeze(-1), 1)
                    #exec_query.append(queries[:, idx])
                    exec_query.append(temp_query[:, idx_in])
                    idx_in += 1
                else:
                    embedding, idx, pre_exec_query = self.embed_query(queries, query_structure[0], idx)
                    exec_query.append(pre_exec_query)
                r_exec_query = []
                for i in range(len(query_structure[-1])):
                    if query_structure[-1][i] == 'n':
                        assert (queries[:, idx_in] == -2).all()
                        r_exec_query.append('n')
                    else:
                        # r_embedding = np.array(self.relation_embeddings)[queries[:, idx]]
                        # r_embedding = torch.index_select(self.relation_embeddings, dim=0, index=queries[:, idx])
                        r_embedding = self.relation_embeddings[temp_query[0, idx_in]]

                        if (i < len(query_structure[-1]) - 1) and query_structure[-1][i + 1] == 'n':
                            embedding, r_argmax = self.relation_projection(embedding, r_embedding, True)
                        else:
                            embedding, r_argmax = self.relation_projection(embedding, r_embedding, False)
                        r_exec_query.append((temp_query[0, idx_in].item(), r_argmax))
                        r_exec_query.append('e')
                    idx_in += 1
                embeddings_out.append(embedding.unsqueeze(0))
                r_exec_query.pop()
                exec_query.append(r_exec_query)
                exec_query.append('e')
        else:
            embedding_list = []
            union_flag = False
            for ele in query_structure[-1]:
                if ele == 'u':
                    union_flag = True
                    query_structure = query_structure[:-1]
                    break
            for i in range(len(query_structure)):
                embedding, idx, pre_exec_query = self.embed_query(queries, query_structure[i], idx)
                embedding_list.append(embedding)
                exec_query.append(pre_exec_query)
            if union_flag:
                embedding = self.union(torch.stack(embedding_list))
                idx += 1
                exec_query.append(['u'])
            else:
                embedding = self.intersection(torch.stack(embedding_list))
            exec_query.append('e')
        
        return torch.cat(embeddings_out, dim=0), idx, exec_query

    def embed_query_ori(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        exec_query = []
        for ele in query_structure[
            -1]:  # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            for p in range(len(queries)):
                temp_query = queries[p]
                if query_structure[0] == 'e':
                    bsz = temp_query.size(0)
                    print(bsz)
                    embedding = torch.zeros(bsz, self.nentity).to(torch.float).to(
                        self.device)  # starting point distribution
                    # embedding.scatter_(-1, queries[:, idx].unsqueeze(-1), 1)
                    embedding.scatter_(-1, temp_query[:, idx].unsqueeze(-1), 1)
                    # exec_query.append(queries[:, idx])
                    exec_query.append(temp_query[:, idx])
                    idx += 1
                else:
                    embedding, idx, pre_exec_query = self.embed_query(queries, query_structure[0], idx)
                    exec_query.append(pre_exec_query)
                r_exec_query = []
                for i in range(len(query_structure[-1])):
                    if query_structure[-1][i] == 'n':
                        assert (queries[:, idx] == -2).all()
                        r_exec_query.append('n')
                    else:
                        # r_embedding = np.array(self.relation_embeddings)[queries[:, idx]]
                        # r_embedding = torch.index_select(self.relation_embeddings, dim=0, index=queries[:, idx])
                        r_embedding = self.relation_embeddings[queries[:, idx]]

                        if (i < len(query_structure[-1]) - 1) and query_structure[-1][i + 1] == 'n':
                            embedding, r_argmax = self.relation_projection(embedding, r_embedding, True)
                        else:
                            embedding, r_argmax = self.relation_projection(embedding, r_embedding, False)
                        r_exec_query.append((queries[:, idx].item(), r_argmax))
                        r_exec_query.append('e')
                    idx += 1
                r_exec_query.pop()
                exec_query.append(r_exec_query)
                exec_query.append('e')
        else:
            embedding_list = []
            union_flag = False
            for ele in query_structure[-1]:
                if ele == 'u':
                    union_flag = True
                    query_structure = query_structure[:-1]
                    break
            for i in range(len(query_structure)):
                embedding, idx, pre_exec_query = self.embed_query(queries, query_structure[i], idx)
                embedding_list.append(embedding)
                exec_query.append(pre_exec_query)
            if union_flag:
                embedding = self.union(torch.stack(embedding_list))
                idx += 1
                exec_query.append(['u'])
            else:
                embedding = self.intersection(torch.stack(embedding_list))
            exec_query.append('e')

        return embedding, idx, exec_query

    def find_ans(self, exec_query, query_structure, anchor):
        ans_structure = self.name_answer_dict[self.query_name_dict[query_structure]]
        return self.backward_ans(ans_structure, exec_query, anchor)

    def backward_ans(self, ans_structure, exec_query, anchor):
        if ans_structure == 'e': # 'e'
            return exec_query, exec_query

        elif ans_structure[0] == 'u': # 'u'
            return ['u'], 'u'
        
        elif ans_structure[0] == 'r': # ['r', 'e', 'r']
            cur_ent = anchor
            ans = []
            for ele, query_ele in zip(ans_structure[::-1], exec_query[::-1]):
                if ele == 'r':
                    r_id, r_argmax = query_ele
                    ans.append(r_id)
                    cur_ent = int(r_argmax[cur_ent])
                elif ele == 'n':
                    ans.append('n')
                else:
                    ans.append(cur_ent)
            return ans[::-1], cur_ent

        elif ans_structure[1][0] == 'r': # [[...], ['r', ...], 'e']
            r_ans, r_ent = self.backward_ans(ans_structure[1], exec_query[1], anchor)
            e_ans, e_ent = self.backward_ans(ans_structure[0], exec_query[0], r_ent)
            ans = [e_ans, r_ans, anchor]
            return ans, e_ent
            
        else: # [[...], [...], 'e']
            ans = []
            for ele, query_ele in zip(ans_structure[:-1], exec_query[:-1]):
                ele_ans, ele_ent = self.backward_ans(ele, query_ele, anchor)
                ans.append(ele_ans)
            ans.append(anchor)
            return ans, ele_ent