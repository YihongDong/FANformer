#!/usr/bin/env python3
"""
简单算术任务测试脚本
用于测试 FANformer 和 StackTrans 在加法任务上的稳定性

任务: 学习 a + b = c 的模式 (0-99范围内的加法)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import argparse
from tqdm import tqdm

from olmo.config import ModelConfig
from olmo.model import OLMo


class ArithmeticDataset(Dataset):
    """简单加法数据集: "a + b = c" """
    
    def __init__(self, num_samples=10000, max_num=99, seed=42):
        random.seed(seed)
        self.samples = []
        self.max_num = max_num
        
        for _ in range(num_samples):
            a = random.randint(0, max_num)
            b = random.randint(0, max_num)
            c = a + b
            self.samples.append((a, b, c))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        a, b, c = self.samples[idx]
        return a, b, c


def create_tokenizer():
    """创建简单的 token 映射"""
    tokens = {
        '<pad>': 0,
        '<bos>': 1,
        '<eos>': 2,
        '+': 3,
        '=': 4,
        ' ': 5,
    }
    for i in range(10):
        tokens[str(i)] = 6 + i
    return tokens


def encode_sample(a, b, c, tokenizer, max_len=20):
    """将 a + b = c 编码为 token 序列"""
    text = f"{a} + {b} = {c}"
    tokens = [tokenizer['<bos>']]
    for char in text:
        if char in tokenizer:
            tokens.append(tokenizer[char])
    tokens.append(tokenizer['<eos>'])
    
    while len(tokens) < max_len:
        tokens.append(tokenizer['<pad>'])
    
    return tokens[:max_len]


def collate_fn(batch, tokenizer, max_len=20):
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


def create_model(model_type='transformer', device='cpu'):
    """创建模型"""
    config = ModelConfig(
        d_model=128,
        n_heads=4,
        n_layers=4,
        mlp_ratio=4,
        vocab_size=16,
        max_sequence_length=32,
        embedding_size=16,
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
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0
    
    pad_id = tokenizer['<pad>']
    
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
            mask = (labels != pad_id)
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()
    
    accuracy = correct / total if total > 0 else 0
    return total_loss / num_batches, accuracy


def main():
    parser = argparse.ArgumentParser(description='测试 FANformer/StackTrans 在算术任务上的稳定性')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['transformer', 'fanformer', 'stacktrans', 'both', 'all'],
                        help='模型类型')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--train_samples', type=int, default=5000, help='训练样本数')
    parser.add_argument('--test_samples', type=int, default=1000, help='测试样本数')
    parser.add_argument('--device', type=str, default='cpu', help='设备')
    args = parser.parse_args()
    
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    
    print(f"使用设备: {args.device}")
    
    tokenizer = create_tokenizer()
    
    train_dataset = ArithmeticDataset(num_samples=args.train_samples, seed=42)
    test_dataset = ArithmeticDataset(num_samples=args.test_samples, seed=123)
    
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
        
        model = create_model(model_type, args.device)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {num_params:,}")
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        
        train_losses = []
        test_losses = []
        test_accs = []
        
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, args.device, tokenizer)
            test_loss, test_acc = evaluate(model, test_loader, args.device, tokenizer)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                      f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        
        results[model_type] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'test_accs': test_accs,
            'final_acc': test_accs[-1],
            'num_params': num_params,
        }
        
        del model
        torch.cuda.empty_cache() if args.device == 'cuda' else None
    
    print(f"\n{'='*60}")
    print("最终结果汇总")
    print(f"{'='*60}")
    print(f"{'模型':<15} {'参数量':<15} {'最终准确率':<15} {'最终Loss':<15}")
    print("-" * 60)
    for model_type, res in results.items():
        print(f"{model_type:<15} {res['num_params']:<15,} {res['final_acc']:<15.4f} {res['test_losses'][-1]:<15.4f}")
    
    print("\n训练完成!")


if __name__ == '__main__':
    main()
