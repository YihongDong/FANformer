#!/usr/bin/env python3
"""
简化版模运算任务测试
使用更小的模数和更长的训练周期来验证模型稳定性
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import argparse

from olmo.config import ModelConfig
from olmo.model import OLMo


class ModularAdditionDataset(Dataset):
    """模运算数据集: (a + b) mod P"""
    
    def __init__(self, P=5, train=True, seed=42):
        random.seed(seed)
        self.P = P
        self.samples = []
        
        all_pairs = [(a, b, (a + b) % P) for a in range(P) for b in range(P)]
        
        if train:
            for a, b, c in all_pairs:
                for _ in range(100):
                    self.samples.append((a, b, c))
        else:
            self.samples = all_pairs * 10
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def create_tokenizer(P=5):
    """创建 token 映射"""
    tokens = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '+': 3, '=': 4}
    for i in range(P * 2):
        tokens[str(i)] = 5 + i
    return tokens


def encode_sample(a, b, c, tokenizer, max_len=8):
    """编码: "<bos> a + b = c <eos>" """
    tokens = [tokenizer['<bos>'], tokenizer[str(a)], tokenizer['+'], 
              tokenizer[str(b)], tokenizer['='], tokenizer[str(c)], tokenizer['<eos>']]
    while len(tokens) < max_len:
        tokens.append(tokenizer['<pad>'])
    return tokens[:max_len]


def collate_fn(batch, tokenizer, max_len=8):
    input_ids = []
    labels = []
    for a, b, c in batch:
        tokens = encode_sample(a, b, c, tokenizer, max_len)
        input_ids.append(tokens[:-1])
        labels.append(tokens[1:])
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


def create_model(model_type='transformer', vocab_size=20, device='cpu'):
    """创建小模型"""
    config = ModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=3,
        mlp_ratio=4,
        vocab_size=vocab_size,
        max_sequence_length=16,
        embedding_size=vocab_size,
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
        stack_dim=8,
    )
    
    model = OLMo(config)
    return model.to(device)


def train_epoch(model, dataloader, optimizer, device, tokenizer):
    model.train()
    total_loss = 0
    num_batches = 0
    pad_id = tokenizer['<pad>']
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        output = model(input_ids)
        logits = output.logits
        
        loss_mask = (labels != pad_id).float()
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none'
        )
        loss = (loss.view_as(labels) * loss_mask).sum() / loss_mask.sum()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, dataloader, device, tokenizer):
    model.eval()
    correct = 0
    total = 0
    eq_id = tokenizer['=']
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            output = model(input_ids)
            logits = output.logits
            preds = logits.argmax(dim=-1)
            
            result_pos = (input_ids == eq_id)
            correct += ((preds == labels) & result_pos).sum().item()
            total += result_pos.sum().item()
    
    return correct / total if total > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all', 
                        choices=['transformer', 'fanformer', 'stacktrans', 'both', 'all'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--P', type=int, default=5, help='模数 (小一点更容易学)')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    
    print(f"模运算任务: (a + b) mod {args.P}")
    print(f"使用设备: {args.device}")
    print(f"所有 {args.P}x{args.P}={args.P**2} 个组合都用于训练和测试")
    
    tokenizer = create_tokenizer(args.P)
    vocab_size = len(tokenizer)
    
    train_dataset = ModularAdditionDataset(P=args.P, train=True, seed=42)
    test_dataset = ModularAdditionDataset(P=args.P, train=False, seed=123)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, tokenizer))
    
    if args.model == 'all':
        model_types = ['transformer', 'fanformer', 'stacktrans', 'both']
    else:
        model_types = [args.model]
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"训练: {model_type.upper()}")
        print(f"{'='*50}")
        
        model = create_model(model_type, vocab_size, args.device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {num_params:,}")
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, args.device, tokenizer)
            test_acc = evaluate(model, test_loader, args.device, tokenizer)
            
            if epoch % 20 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Acc: {test_acc:.4f}")
        
        final_acc = evaluate(model, test_loader, args.device, tokenizer)
        results[model_type] = {'params': num_params, 'acc': final_acc}
        
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"\n{'='*50}")
    print("最终结果")
    print(f"{'='*50}")
    for m, r in results.items():
        print(f"{m:<15} 参数: {r['params']:<10,} 准确率: {r['acc']:.4f}")


if __name__ == '__main__':
    main()
