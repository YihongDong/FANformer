#!/usr/bin/env python3
"""
模运算任务测试脚本
这是 FANformer 论文中的经典测试任务: (a + b) mod P

测试模型对周期性模式的学习能力
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
    
    def __init__(self, P=113, num_samples=10000, seed=42, exclude_pairs=None):
        random.seed(seed)
        self.P = P
        self.samples = []
        self.exclude_pairs = exclude_pairs or set()
        
        all_pairs = [(a, b) for a in range(P) for b in range(P)]
        valid_pairs = [p for p in all_pairs if p not in self.exclude_pairs]
        
        for _ in range(num_samples):
            a, b = random.choice(valid_pairs)
            c = (a + b) % P
            self.samples.append((a, b, c))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def create_tokenizer(P=113):
    """创建 token 映射"""
    tokens = {
        '<pad>': 0,
        '<bos>': 1,
        '<eos>': 2,
        '+': 3,
        '=': 4,
        ' ': 5,
    }
    for i in range(P + P):
        tokens[str(i)] = 6 + i
    return tokens


def encode_sample(a, b, c, tokenizer, max_len=15):
    """编码为 token 序列: "<bos> a + b = c <eos>" """
    tokens = [tokenizer['<bos>'], tokenizer[str(a)], tokenizer['+'], 
              tokenizer[str(b)], tokenizer['='], tokenizer[str(c)], tokenizer['<eos>']]
    
    while len(tokens) < max_len:
        tokens.append(tokenizer['<pad>'])
    
    return tokens[:max_len]


def collate_fn(batch, tokenizer, max_len=15):
    """批处理函数"""
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


def create_model(model_type='transformer', vocab_size=256, device='cpu'):
    """创建模型"""
    config = ModelConfig(
        d_model=256,
        n_heads=8,
        n_layers=6,
        mlp_ratio=4,
        vocab_size=vocab_size,
        max_sequence_length=32,
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
        num_mem_heads=8,
        stack_slots=16,
        stack_dim=16,
    )
    
    model = OLMo(config)
    return model.to(device)


def train_epoch(model, dataloader, optimizer, device, tokenizer):
    """训练一个 epoch"""
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
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction='none'
        )
        loss = (loss.view_as(labels) * loss_mask).sum() / loss_mask.sum()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, dataloader, device, tokenizer):
    """评估模型 - 只计算结果位置的准确率"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0
    
    pad_id = tokenizer['<pad>']
    eq_id = tokenizer['=']
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            output = model(input_ids)
            logits = output.logits
            
            loss_mask = (labels != pad_id).float()
            
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='none'
            )
            loss = (loss.view_as(labels) * loss_mask).sum() / loss_mask.sum()
            
            total_loss += loss.item()
            num_batches += 1
            
            preds = logits.argmax(dim=-1)
            
            result_pos = (input_ids == eq_id)
            correct += ((preds == labels) & result_pos).sum().item()
            total += result_pos.sum().item()
    
    accuracy = correct / total if total > 0 else 0
    return total_loss / num_batches, accuracy


def main():
    parser = argparse.ArgumentParser(description='测试模运算任务')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['transformer', 'fanformer', 'stacktrans', 'both', 'all'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--P', type=int, default=97, help='模数')
    parser.add_argument('--train_samples', type=int, default=8000)
    parser.add_argument('--test_samples', type=int, default=1000)
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
    
    tokenizer = create_tokenizer(args.P)
    vocab_size = len(tokenizer)
    print(f"词表大小: {vocab_size}")
    
    train_dataset = ModularAdditionDataset(P=args.P, num_samples=args.train_samples, seed=42)
    test_dataset = ModularAdditionDataset(P=args.P, num_samples=args.test_samples, seed=123)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    
    if args.model == 'all':
        model_types = ['transformer', 'fanformer', 'stacktrans', 'both']
    else:
        model_types = [args.model]
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"训练模型: {model_type.upper()}")
        print(f"{'='*60}")
        
        model = create_model(model_type, vocab_size, args.device)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {num_params:,}")
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_acc = 0
        train_losses = []
        test_accs = []
        
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, args.device, tokenizer)
            test_loss, test_acc = evaluate(model, test_loader, args.device, tokenizer)
            scheduler.step()
            
            train_losses.append(train_loss)
            test_accs.append(test_acc)
            best_acc = max(best_acc, test_acc)
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                      f"Test Loss: {test_loss:.4f} | Result Acc: {test_acc:.4f}")
        
        results[model_type] = {
            'train_losses': train_losses,
            'test_accs': test_accs,
            'best_acc': best_acc,
            'final_acc': test_accs[-1],
            'num_params': num_params,
        }
        
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"最终结果汇总 (模运算 mod {args.P})")
    print(f"{'='*60}")
    print(f"{'模型':<15} {'参数量':<15} {'最佳准确率':<15} {'最终准确率':<15}")
    print("-" * 60)
    for model_type, res in results.items():
        print(f"{model_type:<15} {res['num_params']:<15,} {res['best_acc']:<15.4f} {res['final_acc']:<15.4f}")
    
    print("\n训练完成!")


if __name__ == '__main__':
    main()
