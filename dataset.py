import numpy as np
import torch

from torch.utils.data import Dataset
flatten=lambda l: sum(map(flatten, l),[]) if isinstance(l,tuple) else [l]

class TestDataset_ori(Dataset):
    def __init__(self, queries, nentity, nrelation):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        return flatten(query), query, query_structure

    @staticmethod
    def collate_fn(data):
        query = [_[0] for _ in data]
        query_unflatten = [_[1] for _ in data]
        query_structure = [_[2] for _ in data]
        return query, query_unflatten, query_structure



class TestDataset(Dataset):
    def __init__(self, queries, nentity, nrelation, quads_to_filter, rule_num=8):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation
        self.rule_num = rule_num
        self.quads = set(quads_to_filter)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        guidance = self.queries[idx][1]
        #confs = self.queries[idx][2]
        '''print(guidance)
        guidance = np.array(guidance, dtype=object)
        fired_rules = guidance[:, 0]
        fired_rules = [flatten(i) for i in fired_rules]
        query_structure = guidance[:,1]'''
        s, r, o, t = query
        tmp = [(0, rand_tail) if (s, r, rand_tail, t) not in self.quads
               else (-10000, o) for rand_tail in range(self.nentity)]
        tmp[o] = (0, o)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        return  query, guidance, filter_bias

    @staticmethod
    def collate_fn(data):
        #negative_sample = torch.stack([_[0] for _ in data], dim=0)
        query = [_[0] for _ in data]

        query_unflatten = []
        query_structure = []
        query_pointer = []
        query_heads = []
        query_confs = []
        query_constraints = []
        for i in data:
            guidance = i[1]
            temp_guid = guidance[:40]
            #query_confs += temp_confs
            for j in temp_guid:
                query_unflatten.append(flatten(j[0]))
                query_structure.append(j[1])
                query_heads.append(i[0])
                query_confs.append(j[2])
                query_constraints.append(j[3])
            query_pointer.append(len(temp_guid))
        filter_bias = [_[2] for _ in data]
        filter_bias = torch.stack(filter_bias, dim=0)
        '''query_unflatten = [flatten(_[2][0]) for _ in data]
        query_structure = [_[2][1] for _ in data]
        neg_piece = data[0][0]
        negative_sample = torch.stack([neg_piece] * sum(query_pointer), dim=0)'''
        return query, query_unflatten, query_structure, query_pointer, filter_bias, query_heads, query_confs, query_constraints