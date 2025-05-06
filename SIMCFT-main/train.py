import argparse
import torch
import torch.utils.data as tud
from torch.utils.data import DataLoader
import numpy as np
import json

from SIMCFT_415 import (
    MetricLearningDataset, collate_fn,
    SeqHardSampler, GlobalHardSampler,
    SIMCFT
)
import utils
import parameters

timer = utils.Timer()

def log_to_file(message, log_file='data/train_log/train_result'):
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def train_SIMCFT(args):
    log_file = 'data/train_log/train_result'

    train_dataset_path = args.train_dataset
    validate_dataset_path = args.validate_dataset
    test_dataset_path = args.test_dataset
    batch_size = args.batch_size
    pretrained_embedding_file = args.pretrained_embedding
    emb_size = args.embedding_size
    learning_rate = args.learning_rate
    epochs = args.epoch_num
    cp = args.checkpoint
    vocab_size = args.vocab_size
    loss_func_name = args.loss_func
    sampler_name = args.sampler
    triplet_num = args.triplet_num
    neg_rate = args.neg_rate
    dataset_size = args.dataset_size
    encoder_layers = args.encoder_layers
    min_len = args.min_len
    cumulative_iters = args.cumulative_iters
    max_len = args.max_len
    heads = args.heads
    attention_gru_hidden_dim = args.attention_gru_hidden_dim
    device = torch.device(args.device)

    timer.tik("prepare data")
    train_data = MetricLearningDataset(
        file_train=train_dataset_path,
        triplet_num=triplet_num,
        min_len=min_len, max_len=max_len,
        dataset_size=dataset_size,
        neg_rate=neg_rate
    )
    if sampler_name == "seq_hard":
        sampler = SeqHardSampler(train_data, batch_size)
        train_loader = tud.DataLoader(train_data, batch_size=batch_size,
                                      collate_fn=collate_fn, sampler=sampler)
    else:
        train_loader = tud.DataLoader(train_data, batch_size=batch_size,
                                      collate_fn=collate_fn, shuffle=True)

    validate_data = MetricLearningDataset(
        file_train=validate_dataset_path,
        triplet_num=triplet_num,
        min_len=0, max_len=99999,
        dataset_size=None,
        neg_rate=neg_rate
    )
    validate_loader = tud.DataLoader(validate_data, batch_size=len(validate_data), collate_fn=collate_fn)

    test_data = MetricLearningDataset(
        file_train=test_dataset_path,
        triplet_num=triplet_num,
        min_len=0, max_len=99999,
        dataset_size=None,
        neg_rate=neg_rate
    )
    test_loader = tud.DataLoader(test_data, batch_size=len(test_data), collate_fn=collate_fn)

    log_to_file(f"prepare data done, {timer.tok('prepare data')} seconds", log_file)

    timer.tik("init model")
    pre_emb = None
    if pretrained_embedding_file:
        pre_emb_array = np.load(pretrained_embedding_file)
        pre_emb = torch.FloatTensor(pre_emb_array)

    model = SIMCFT(
        vocab_size=vocab_size,
        emb_size=emb_size,
        heads=heads,
        encoder_layers=encoder_layers,
        attention_gru_hidden_dim=attention_gru_hidden_dim,
        pre_emb=pre_emb,
        tau=0.1
    ).to(device)

    model.mean_x = train_data.meanx
    model.mean_y = train_data.meany
    model.std_x  = train_data.stdx
    model.std_y  = train_data.stdy

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    epoch_start = 0

    if loss_func_name == "seq_hard":
        loss_func = model.calculate_loss_seq_sampler
    elif loss_func_name == "infoNCE":
        loss_func = model.calculate_loss_infoNCE
    else:
        loss_func = model.calculate_loss_vanilla

    if cp is not None:
        cp_data = torch.load(cp)
        if 'model' in cp_data:
            model.load_state_dict(cp_data['model'])
        if 'optimizer' in cp_data:
            optimizer.load_state_dict(cp_data['optimizer'])
        if 'epoch' in cp_data:
            epoch_start = cp_data['epoch'] + 1

    log_to_file(f"init model done, {timer.tok('init model')} seconds", log_file)

    best_rank = 99999.0
    for epoch in range(epoch_start, epochs):
        for batch_idx, batch_data in enumerate(train_loader):
            (anchor, anchor_lens,
             pos, pos_lens,
             neg, neg_lens,
             trajs_a, trajs_a_lens,
             trajs_p, trajs_p_lens,
             trajs_n, trajs_n_lens,
             anchor_idxs, pos_idxs, neg_idxs,
             sim_pos, sim_neg, sim_matrix_a) = batch_data

            anchor = anchor.to(device)
            pos    = pos.to(device)
            neg    = neg.to(device)
            trajs_a = trajs_a.to(device)
            trajs_p = trajs_p.to(device)
            trajs_n = trajs_n.to(device)
            sim_pos = sim_pos.to(device)
            sim_neg = sim_neg.to(device)
            sim_matrix_a = sim_matrix_a.to(device)

            result = loss_func(
                anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                trajs_a, trajs_a_lens, trajs_p, trajs_p_lens,
                trajs_n, trajs_n_lens, anchor_idxs, pos_idxs, neg_idxs,
                sim_pos, sim_neg, sim_matrix_a
            )
            if isinstance(result, tuple):
                loss = result[0]
            else:
                loss = result

            loss = loss / cumulative_iters
            loss.backward()

            if batch_idx % cumulative_iters == 0:
                optimizer.step()
                optimizer.zero_grad()

            log_str = f"[epoch {epoch} batch {batch_idx}] loss={loss.item():.4f}, lambda={model.lamb.item():.3f}"
            print(log_str)
            log_to_file(log_str, log_file)

        rank_test, hr_10_test, hr_20_test, r10_50_test, r10_100_test, _ = model.evaluate(
            test_loader, device, tri_num=triplet_num
        )
        msg = f"[epoch {epoch}] test rank={rank_test:.4f}, hr10={hr_10_test:.4f}, hr20={hr_20_test:.4f}"
        print(msg)
        log_to_file(msg, log_file)

        if rank_test < best_rank:
            best_rank = rank_test
            cp_data = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(cp_data, f"model/best_model.pth")
            log_to_file("[SAVE] new best model", log_file)

def main():
    parser = argparse.ArgumentParser(description="train SIMCFT")

    parser.add_argument('--model', default='SIMCFT', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--cumulative_iters', default=1, type=int)
    parser.add_argument('--epoch_num', default=50, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--device', default="cuda", type=str)

    parser.add_argument('--train_dataset', default="data/train/train_dataset.json", type=str)
    parser.add_argument('--validate_dataset', default="data/valid/valid_dataset.json", type=str)
    parser.add_argument('--test_dataset', default="data/test/test_frechet.json", type=str)

    parser.add_argument('--pretrained_embedding', default=None, type=str)
    parser.add_argument('--loss_func', default="infoNCE", type=str)
    parser.add_argument('--sampler', default="vanilla", type=str)

    parser.add_argument('--embedding_size', default=64, type=int)
    parser.add_argument('--vocab_size', default=31925, type=int)
    parser.add_argument('--dataset_size', default=None, type=int)
    parser.add_argument('--min_len', default=0, type=int)
    parser.add_argument('--max_len', default=99999, type=int)
    parser.add_argument('--triplet_num', default=4, type=int)
    parser.add_argument('--neg_rate', default=4, type=int)

    parser.add_argument('--attention_gru_hidden_dim', default=64, type=int)
    parser.add_argument('--encoder_layers', default=4, type=int)
    parser.add_argument('--heads', default=4, type=int)

    args = parser.parse_args()
    train_SIMCFT(args)

if __name__ == "__main__":
    main()
