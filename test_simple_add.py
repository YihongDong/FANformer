#!/usr/bin/env python3
"""
简单加法任务 - 验证 FANformer/StackTrans 训练稳定性
任务: 学习 a + b = c (0-9 范围内)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import argparse

from olmo.config import ModelConfig
from olmo.model import OLMo


class SimpleAddDataset(Dataset):
    """简单加法: a + b = c, 其中 a,b ∈ [0,9]"""
    
    def __init__(self, train=True, seed=42):
        random.seed(seed)
        self.samples = []
        
        for a in range(10):
            for b in range(10):
                c = a + b
                repeat = 50 if train else 10
                for _ in range(repeat):
                    self.samples.append((a, b, c))
        
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


VOCAB = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '+': 3, '=': 4}
for i in range(20):
    VOCAB[str(i)] = 5 + i


def encode(a, b, c, max_len=10):
    tokens = [VOCAB['<bos>'], VOCAB[str(a)], VOCAB['+'], 
              VOCAB[str(b)], VOCAB['='], VOCAB[str(c)], VOCAB['<eos>']]
    while len(tokens) < max_len:
        tokens.append(VOCAB['<pad>'])
    return tokens[:max_len]


def collate_fn(batch):
    input_ids, labels = [], []
    for a, b, c in batch:
        tokens = encode(a, b, c)
        input_ids.append(tokens[:-1])
        labels.append(tokens[1:])
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


def create_model(model_type, device):
    config = ModelConfig(
        d_model=128,
        n_heads=4,
        n_layers=4,
        mlp_ratio=4,
        vocab_size=len(VOCAB),
        max_sequence_length=16,
        embedding_size=len(VOCAB),
        weight_tying=True,
        include_bias=True,
        flash_attention=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        rope=True,
        init_device=device,
        
        use_ATF=(model_type in ['fanformer', 'both']),
        p_ratio=0.25,
        
        use_stack_memory=(model_type in ['stacktrans', 'both']),
        num_mem_heads=4,
        stack_slots=8,
        stack_dim=16,
    )
    return OLMo(config).to(device)


def train_and_eval(model, train_loader, test_loader, epochs, lr, device):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    pad_id = VOCAB['<pad>']
    eq_id = VOCAB['=']
    
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids).logits
            
            mask = (labels != pad_id).float()
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none'
            )
            loss = (loss.view_as(labels) * mask).sum() / mask.sum()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                preds = model(input_ids).logits.argmax(-1)
                
                pos = (input_ids == eq_id)
                correct += ((preds == labels) & pos).sum().item()
                total += pos.sum().item()
        
        acc = correct / total if total > 0 else 0
        avg_loss = total_loss / len(train_loader)
        history.append((avg_loss, acc))
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
    
    return history[-1][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='all', choices=['transformer', 'fanformer', 'stacktrans', 'both', 'all'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"简单加法任务: a + b = c (a,b ∈ [0,9])")
    print(f"设备: {args.device}")
    
    train_loader = DataLoader(SimpleAddDataset(train=True), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(SimpleAddDataset(train=False), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    models = ['transformer', 'fanformer', 'stacktrans', 'both'] if args.model == 'all' else [args.model]
    results = {}
    
    for m in models:
        print(f"\n{'='*40}")
        print(f"{m.upper()}")
        print(f"{'='*40}")
        
        model = create_model(m, args.device)
        params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {params:,}")
        
        acc = train_and_eval(model, train_loader, test_loader, args.epochs, args.lr, args.device)
        results[m] = {'params': params, 'acc': acc}
        
        del model
    
    print(f"\n{'='*40}")
    print("汇总")
    print(f"{'='*40}")
    for m, r in results.items():
        status = "OK" if r['acc'] > 0.95 else "需更多训练" if r['acc'] > 0.5 else "有问题"
        print(f"{m:<15} 准确率: {r['acc']:.4f}  [{status}]")


if __name__ == '__main__':
    main()
