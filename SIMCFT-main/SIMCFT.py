import torch
import torch.utils.data as tud
from torch.utils.data.sampler import Sampler
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch import nn
import numpy as np
import json
import math
import random
import utils

timer = utils.Timer()

class MetricLearningDataset(tud.Dataset):
    def __init__(self, file_train, triplet_num, min_len, max_len, dataset_size=None, neg_rate=10):

        self.triplet_num = triplet_num
        self.min_len = min_len
        self.max_len = max_len
        self.dataset_size = dataset_size
        self.neg_rate = neg_rate

        self.trajs = None             
        self.original_trajs = None    
        self.dis_matrix = None
        self.sorted_index = None
        self.sim_matrix = None
        self.loss_map = None
       
        with open(file_train, 'r') as f:
            train_dict = json.load(f)
        
        self.data_prepare(train_dict)

    def data_prepare(self, train_dict):
       
        x, y = [], []
        for origin_traj in train_dict["origin_trajs"]:
            for row in origin_traj:
                lon = row[0]
                lat = row[1]
                x.append(lon)
                y.append(lat)

        meanx, meany = np.mean(x), np.mean(y)
        stdx, stdy   = np.std(x), np.std(y)
        self.meanx, self.meany, self.stdx, self.stdy = meanx, meany, stdx, stdy

        
        trajs = []
        original_trajs = []
        used_idxs = []
        for idx, idx_traj in enumerate(train_dict["trajs"]):
            if self.min_len < len(idx_traj) < self.max_len:
                trajs.append(idx_traj)

                
                raw_traj = train_dict["origin_trajs"][idx]
                norm_2d_traj = []
                for row in raw_traj:
                    lon = row[0]
                    lat = row[1]
                    norm_lon = (lon - meanx)/stdx
                    norm_lat = (lat - meany)/stdy
                    norm_2d_traj.append([norm_lon, norm_lat])

                original_trajs.append(norm_2d_traj)
                used_idxs.append(idx)

        if self.dataset_size is None:
            self.dataset_size = len(used_idxs)
        else:
            self.dataset_size = min(self.dataset_size, len(used_idxs))

        used_idxs = used_idxs[:self.dataset_size]

        self.trajs = np.array(trajs[:self.dataset_size], dtype=object)
        self.original_trajs = np.array(original_trajs[:self.dataset_size], dtype=object)

       
        dist_mat = np.array(train_dict["dis_matrix"])
        dist_mat = dist_mat[used_idxs, :][:, used_idxs]
        self.dis_matrix = dist_mat

        
        self.sorted_index = np.argsort(self.dis_matrix, axis=1)
        a = 20
        self.sim_matrix = np.exp(-a * self.dis_matrix)

        
        self.loss_map = torch.zeros(self.dataset_size, self.dataset_size)
        for i in range(self.dataset_size):
            self.loss_map[i, i] = -1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        
        anchor = self.trajs[idx]

        
        pos_idx_list = self.sorted_index[idx][:self.triplet_num+1].tolist()
        if idx in pos_idx_list:
            pos_idx_list.remove(idx)
        else:
            pos_idx_list = pos_idx_list[:self.triplet_num]
        positive = self.trajs[pos_idx_list]

        
        negative_idx = np.random.choice(
            self.sorted_index[idx][self.triplet_num:],
            self.triplet_num * self.neg_rate,
            replace=False
        ).tolist()
        negative = self.trajs[negative_idx]

        
        trajs_a = self.original_trajs[idx]
        trajs_p = self.original_trajs[pos_idx_list]
        trajs_n = self.original_trajs[negative_idx]

        return (
            anchor,
            positive,
            negative,
            trajs_a,
            trajs_p,
            trajs_n,
            idx,
            pos_idx_list,
            negative_idx,
            self.sim_matrix[idx, pos_idx_list],
            self.sim_matrix[idx, negative_idx],
            self.sim_matrix[idx, :]
        )


def collate_fn(train_data):
    batch_size = len(train_data)
    anchor = [torch.tensor(d[0]) for d in train_data]
    anchor_lens = [len(a) for a in anchor]
    anchor = rnn_utils.pad_sequence(anchor, batch_first=True, padding_value=-1)

    pos = []
    for subp in [d[1] for d in train_data]:
        pos.extend(subp)
    pos = [torch.tensor(p) for p in pos]
    pos_lens = [len(x) for x in pos]
    pos = rnn_utils.pad_sequence(pos, batch_first=True, padding_value=-1)

    neg = []
    for subn in [d[2] for d in train_data]:
        neg.extend(subn)
    neg = [torch.tensor(n) for n in neg]
    neg_lens = [len(x) for x in neg]
    neg = rnn_utils.pad_sequence(neg, batch_first=True, padding_value=-1)

    trajs_a = [torch.tensor(np.array(d[3]), dtype=torch.float32) for d in train_data]
    trajs_a_lens = [len(ta) for ta in trajs_a]
    trajs_a = rnn_utils.pad_sequence(trajs_a, batch_first=True, padding_value=0)

    trajs_p = []
    for subp in [d[4] for d in train_data]:
        trajs_p.extend(subp)
    trajs_p = [torch.tensor(np.array(tp), dtype=torch.float32) for tp in trajs_p]
    trajs_p_lens = [len(tp) for tp in trajs_p]
    trajs_p = rnn_utils.pad_sequence(trajs_p, batch_first=True, padding_value=0)

    trajs_n = []
    for subn in [d[5] for d in train_data]:
        trajs_n.extend(subn)
    trajs_n = [torch.tensor(np.array(tn), dtype=torch.float32) for tn in trajs_n]
    trajs_n_lens = [len(tn) for tn in trajs_n]
    trajs_n = rnn_utils.pad_sequence(trajs_n, batch_first=True, padding_value=0)

    anchor_idxs = torch.tensor([d[6] for d in train_data], dtype=torch.long)

    pos_idxs = []
    for subp in [d[7] for d in train_data]:
        pos_idxs.extend(subp)
    pos_idxs = torch.tensor(pos_idxs, dtype=torch.long)

    neg_idxs = []
    for subn in [d[8] for d in train_data]:
        neg_idxs.extend(subn)
    neg_idxs = torch.tensor(neg_idxs, dtype=torch.long)

    sim_pos = torch.tensor(np.array([d[9] for d in train_data]), dtype=torch.float32)
    sim_neg = torch.tensor(np.array([d[10] for d in train_data]), dtype=torch.float32)
    sim_matrix_a = torch.tensor(np.array([d[11] for d in train_data]), dtype=torch.float32)
    sim_matrix_a = sim_matrix_a[:, anchor_idxs]

    return (
        anchor, anchor_lens,
        pos, pos_lens,
        neg, neg_lens,
        trajs_a, trajs_a_lens,
        trajs_p, trajs_p_lens,
        trajs_n, trajs_n_lens,
        anchor_idxs, pos_idxs, neg_idxs,
        sim_pos, sim_neg, sim_matrix_a
    )


class SeqHardSampler(Sampler):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        indices = []
        loss_map_copy = self.data.loss_map.clone()
        start_seq = [i for i in range(loss_map_copy.shape[0])]
        random.shuffle(start_seq)
        while start_seq:
            idx = start_seq.pop()
            indices.append(idx)
            if len(start_seq) < self.batch_size:
                indices.extend(start_seq)
                start_seq.clear()
                break
            losses, indexs = torch.topk(loss_map_copy[idx], k=self.batch_size-1, dim=0)
            loss_map_copy[:, indexs] = -1
            loss_map_copy[:, idx] = -1
            for index in indexs:
                start_seq.remove(index)
            indices.extend(indexs.tolist())
        indices = torch.LongTensor(indices)
        return iter(indices)

    def __len__(self):
        return len(self.data)


class GlobalHardSampler(Sampler):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        datasize = self.data.dataset_size
        all_indices = list(range(datasize))
        
        return iter(all_indices)

    def __len__(self):
        return len(self.data)



class DynamicGatedGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(DynamicGatedGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=0.1)
        self.feature_interaction = nn.Linear(hidden_dim, hidden_dim)
        self.dynamic_weighting = nn.Linear(hidden_dim, 1)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, trajs, trajs_lens):
       
        trajs_pack = rnn_utils.pack_padded_sequence(trajs, trajs_lens, batch_first=True, enforce_sorted=False)
        gru_out, _ = self.gru(trajs_pack)
        gru_out, _ = rnn_utils.pad_packed_sequence(gru_out, batch_first=True)

        
        enhanced_features = self.feature_interaction(gru_out)
        dynamic_weight = torch.sigmoid(self.dynamic_weighting(enhanced_features))
        enhanced_features = gru_out + dynamic_weight * enhanced_features

        
        gate_val = torch.sigmoid(self.gate(enhanced_features))
        updated_hidden_state = gate_val * enhanced_features + (1 - gate_val) * gru_out

        updated_hidden_state = self.dropout(updated_hidden_state)
        updated_hidden_state = self.layer_norm(updated_hidden_state + gru_out)

        
        context_vector = torch.sum(updated_hidden_state, dim=1)
        return context_vector



class NoiseFilteringLayer(nn.Module):
    def __init__(self, input_dim):
        super(NoiseFilteringLayer, self).__init__()
        self.noise_weights = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
       
        noise_weights = self.noise_weights(x)  
        filtered_output = noise_weights * x
        return filtered_output


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.0, pe_max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(pe_max_len, emb_size)
        position = torch.arange(0, pe_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        x = x.transpose(0,1)  # -> [T,B,D]
        x = x + self.pe[:x.size(0), :]
        x = x.transpose(0,1)  # -> [B,T,D]
        return self.dropout(x)



class SIMCFT(nn.Module):
    def __init__(self, vocab_size, emb_size, heads=8, encoder_layers=1,
                 attention_gru_hidden_dim=128, pre_emb=None, t2g=None, tau=0.1):
        super(SIMCFT, self).__init__()
        self.lamb = nn.Parameter(torch.FloatTensor(1), requires_grad=True)  
        nn.init.constant_(self.lamb, 0.5)
        self.tau = tau
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)  
        nn.init.constant_(self.alpha, 1)

        if pre_emb is not None:
            self.embedding = nn.Embedding(vocab_size, emb_size).from_pretrained(pre_emb, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        self.position_encoding = PositionalEncoding(emb_size, dropout=0.1)

        
        self.noise_filtering_layer = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.Sigmoid()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            emb_size, heads, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        
        self.attention_gru = DynamicGatedGRU(2, attention_gru_hidden_dim, num_layers=encoder_layers)

        self.t2g = t2g
        self.mean_x = None
        self.mean_y = None
        self.std_x = None
        self.std_y = None

    def forward(self, x, trajs, trajs_lens):
        
        mask_x = (x == -1)
        x = x.clamp_min(0)
        emb_x = self.embedding(x)
        emb_x = self.position_encoding(emb_x)

        
        emb_x = self.noise_filtering_layer(emb_x)  

        
        encoder_out = self.encoder(emb_x, src_key_padding_mask=mask_x)
        
        valid_mask = (~mask_x).unsqueeze(2).float()
        sum_x = torch.sum(encoder_out * valid_mask, dim=1)
        valid_len = torch.sum(valid_mask, dim=1).clamp_min(1e-9)
        spatial_feat = sum_x / valid_len
        spatial_feat = F.normalize(spatial_feat, p=2, dim=-1)

        
        status_feat = self.attention_gru(trajs, trajs_lens)
        status_feat = F.normalize(status_feat, p=2, dim=-1)

        
        output = self.lamb * spatial_feat + (1 - self.lamb)*status_feat
        return output

    
    def calculate_loss_infoNCE(self, anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                               trajs_a, trajs_a_lens, trajs_p, trajs_p_lens,
                               trajs_n, trajs_n_lens,
                               anchor_idxs, pos_idxs, neg_idxs,
                               sim_pos, sim_neg, sim_matrix_a):
       
        z_a = self.forward(anchor, trajs_a, anchor_lens)   
        z_p = self.forward(pos, trajs_p, pos_lens)        
        z_n = self.forward(neg, trajs_n, neg_lens)         

        
        B = z_a.size(0)
        pos_num = z_p.size(0) // B
        neg_num = z_n.size(0) // B

        z_p = z_p.view(B, pos_num, -1)  
        z_n = z_n.view(B, neg_num, -1)  

        z_a_unsq = z_a.unsqueeze(1)     

        sim_p = torch.exp(- self.alpha * torch.norm(z_a_unsq - z_p, dim=2))  
        sim_n = torch.exp(- self.alpha * torch.norm(z_a_unsq - z_n, dim=2))  

        
        sp_scaled = sim_p / self.tau  
        sn_scaled = sim_n / self.tau  

       
        numerator   = torch.sum(torch.exp(sp_scaled), dim=1, keepdim=True)      
        denominator = numerator + torch.sum(torch.exp(sn_scaled), dim=1, keepdim=True)  

        loss = -torch.log(numerator / denominator)
        return loss.mean()

    

    def calculate_loss_vanilla(self, anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                               trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                               anchor_idxs, pos_idxs, neg_idxs, sim_pos, sim_neg, sim_matrix_a, *args):
        batch_size = anchor.size(0)
        pos_num = pos.shape[0] // batch_size
        device = anchor.device
        output_a = self.forward(anchor, trajs_a, trajs_a_lens)
        output_p = self.forward(pos, trajs_p, trajs_p_lens)
        sim_p = torch.exp(-self.alpha * torch.norm(output_a.repeat(pos_num, 1) - output_p, dim=1)).reshape(batch_size, -1)
        sim_a = torch.exp(-self.alpha * torch.norm(output_a.unsqueeze(1)-output_a, dim=2)).reshape(batch_size, -1)
        w_p = torch.softmax(torch.ones(pos_num)/torch.arange(1, pos_num+1).float(), dim=0).to(device)
        loss_p = torch.sum(w_p * (sim_p - sim_pos)**2, dim=1)
        loss_n = torch.sum((torch.relu(sim_a - sim_matrix_a))**2, dim=1)
        loss = (loss_p + loss_n).mean()
        return loss, loss_p.mean(), loss_n.mean()


    def calculate_loss_vanilla_v2(self, anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                                  trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                                  anchor_idxs, pos_idxs, neg_idxs, sim_pos, sim_neg, sim_matrix_a):
        batch_size = anchor.size(0)
        pos_num = pos.shape[0] // batch_size
        neg_num = neg.shape[0] // batch_size
        device = anchor.device
        output_a = self.forward(anchor, trajs_a, trajs_a_lens)
        output_p = self.forward(pos, trajs_p, trajs_p_lens)
        sim_p = torch.exp(-self.alpha * torch.norm(output_a.repeat(pos_num, 1) - output_p, dim=1)).reshape(batch_size, -1)
        output_n = self.forward(neg, trajs_n, trajs_n_lens)
        sim_n = torch.exp(-self.alpha * torch.norm(output_a.repeat(neg_num, 1) - output_n, dim=1)).reshape(batch_size, -1)
        w_p = torch.softmax(torch.ones(pos_num)/torch.arange(1, pos_num+1).float(), dim=0).to(device)
        w_n = torch.softmax(torch.ones(neg_num).float(), dim=0).to(device)
        loss_p = torch.sum(w_p * (sim_p - sim_pos)**2, dim=1)
        loss_n = torch.sum(w_n * (torch.relu(sim_n - sim_neg))**2, dim=1)
        loss = (loss_p + loss_n).mean()
        return loss

    def calculate_loss_seq_sampler(self, anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                                   trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                                   anchor_idxs, pos_idxs, neg_idxs, sim_pos, sim_neg, sim_matrix_a, loss_map):
        batch_size = anchor.size(0)
        pos_num = pos.shape[0] // batch_size
        device = anchor.device
        output_a = self.forward(anchor, trajs_a, trajs_a_lens)
        output_p = self.forward(pos, trajs_p, trajs_p_lens)
        sim_p = torch.exp(-self.alpha * torch.norm(output_a.repeat(pos_num, 1) - output_p, dim=1)).reshape(batch_size, -1)
        sim_a = torch.exp(-self.alpha * torch.norm(output_a.unsqueeze(1)-output_a, dim=2)).reshape(batch_size, -1)
        w_p = torch.softmax(torch.ones(pos_num)/torch.arange(1, pos_num+1).float(), dim=0).to(device)
        loss_p = torch.sum(w_p * (sim_p - sim_pos)**2, dim=1)
        loss_n = torch.relu(sim_a - sim_matrix_a)
        loss_n_copy = loss_n.detach().cpu()
        idxs_diag = torch.arange(0,batch_size)
        loss_n_copy[idxs_diag,idxs_diag] = -1
        mask = loss_map[anchor_idxs]
        mask[:,anchor_idxs] = loss_n_copy
        loss_map[anchor_idxs] = mask
        loss_n = torch.sum(loss_n, dim=1)
        loss = loss_p.mean() + loss_n.mean()
        return loss, loss_p.mean(), loss_n.mean()

    def calculate_loss_hard_miner(self):
        pass

    def evaluate(self, validate_loader, device, tri_num):
        self.eval()
        ranks = []
        hit_ratios_10 = []
        hit_ratios_20 = []
        ratios10_50 = []
        ratios10_100 = []
        pca_x = []

        with torch.no_grad():
            for (anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                 trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                 anchor_idxs, pos_idxs, neg_idxs, sim_pos, sim_neg, sim_matrix_a) in validate_loader:

                
                anchor = anchor.to(device)
                trajs_a = trajs_a.to(device)
                anchor_lens = anchor_lens  

                
                output_a = self.forward(anchor, trajs_a, anchor_lens)
                

                
                bsz = 300
                sim_matrixs = []
                B = anchor.shape[0]
                for i in range((B // bsz) + 1):
                    lb = i * bsz
                    ub = min((i + 1) * bsz, B)
                    if lb >= ub:
                        break
                    
                    anchor_sub = anchor[lb:ub].to(device)
                    trajs_a_sub = trajs_a[lb:ub].to(device)
                    lens_sub = anchor_lens[lb:ub]

                    output_b = self.forward(anchor_sub, trajs_a_sub, lens_sub)
                   
                    s = torch.exp(- torch.norm(
                        output_a.unsqueeze(1) - output_b.unsqueeze(0), dim=-1
                    ))
                    
                    sim_matrixs.append(s)

                sim_matrix = torch.cat(sim_matrixs, dim=1).cpu().numpy()
                
                sorted_index = np.argsort(-sim_matrix, axis=1)
                sorted_index = sorted_index[:, 1:]  

                
                for i in range(B):
                    row_pos = pos_idxs[i].cpu().numpy()
                    
                    avg_rank = 0
                    if len(row_pos) > 0:
                       
                        for idxx in row_pos:
                            arr = np.argwhere(sorted_index[i] == idxx)
                            if len(arr) > 0:
                                avg_rank += arr[0][0]
                        avg_rank /= len(row_pos)
                    
                    ranks.append(avg_rank)
                    

        self.train()
        
        rank = np.mean(ranks)
        hr_10 = np.mean(hit_ratios_10)
        hr_20 = np.mean(hit_ratios_20)
        r10_50 = np.mean(ratios10_50)
        r10_100 = np.mean(ratios10_100)
        return rank, hr_10, hr_20, r10_50, r10_100, pca_x

