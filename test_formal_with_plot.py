#!/usr/bin/env python3
"""
Formal Language Tasks with Training Visualization
带训练过程可视化的 Formal Language 任务测试
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import argparse
import matplotlib.pyplot as plt
import os

from olmo.config import ModelConfig
from olmo.model import OLMo


# ============================================================================
# Datasets
# ============================================================================

class ParityCheckDataset(Dataset):
    def __init__(self, min_len=1, max_len=20, num_samples=5000, seed=42):
        random.seed(seed)
        self.samples = []
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            s = ''.join(random.choice('ab') for _ in range(length))
            label = s.count('b') % 2 == 0
            self.samples.append((s, label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


class CycleNavigationDataset(Dataset):
    def __init__(self, min_len=1, max_len=20, num_samples=5000, seed=42):
        random.seed(seed)
        self.samples = []
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            moves = [random.randint(0, 2) for _ in range(length)]
            position = 0
            for m in moves:
                if m == 1: position = (position + 1) % 5
                elif m == 2: position = (position - 1) % 5
            self.samples.append((moves, position))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


class ReverseStringDataset(Dataset):
    def __init__(self, min_len=1, max_len=10, num_samples=5000, seed=42, vocab_size=5):
        random.seed(seed)
        self.samples = []
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            s = [random.randint(0, vocab_size-1) for _ in range(length)]
            self.samples.append((s, s[::-1]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


class ModularArithmeticDataset(Dataset):
    def __init__(self, max_depth=3, num_samples=5000, seed=42):
        random.seed(seed)
        self.samples = []
        self.ops = ['+', '-', '*']
        for _ in range(num_samples):
            expr, value = self._generate_expr(random.randint(1, max_depth))
            self.samples.append((expr, value % 5))
    
    def _generate_expr(self, depth):
        if depth == 0 or random.random() < 0.3:
            val = random.randint(0, 4)
            return str(val), val
        left_expr, left_val = self._generate_expr(depth - 1)
        right_expr, right_val = self._generate_expr(depth - 1)
        op = random.choice(self.ops)
        if op == '+': result = (left_val + right_val) % 5
        elif op == '-': result = (left_val - right_val) % 5
        else: result = (left_val * right_val) % 5
        return f"({left_expr}{op}{right_expr})", result
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# ============================================================================
# Tokenizers
# ============================================================================

def create_binary_tokenizer():
    return {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<sep>': 3, 'a': 4, 'b': 5, 'T': 6, 'F': 7}

def create_cycle_tokenizer():
    return {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<sep>': 3, '0': 4, '1': 5, '2': 6,
            'r0': 7, 'r1': 8, 'r2': 9, 'r3': 10, 'r4': 11}

def create_reverse_tokenizer(vocab_size=5):
    tok = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<sep>': 3}
    for i in range(vocab_size): tok[f'v{i}'] = 4 + i
    return tok

def create_modular_tokenizer():
    return {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<sep>': 3,
            '0': 4, '1': 5, '2': 6, '3': 7, '4': 8,
            '+': 9, '-': 10, '*': 11, '(': 12, ')': 13,
            'r0': 14, 'r1': 15, 'r2': 16, 'r3': 17, 'r4': 18}


# ============================================================================
# Collate Functions
# ============================================================================

def collate_binary(batch, tokenizer, max_len=30):
    input_ids, labels = [], []
    for s, label in batch:
        tokens = [tokenizer['<bos>']] + [tokenizer[c] for c in s if c in tokenizer]
        tokens += [tokenizer['<sep>'], tokenizer['T'] if label else tokenizer['F'], tokenizer['<eos>']]
        tokens = (tokens + [tokenizer['<pad>']] * max_len)[:max_len]
        input_ids.append(tokens[:-1])
        labels.append(tokens[1:])
    return {'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels)}

def collate_cycle(batch, tokenizer, max_len=30):
    input_ids, labels = [], []
    for moves, result in batch:
        tokens = [tokenizer['<bos>']] + [tokenizer[str(m)] for m in moves]
        tokens += [tokenizer['<sep>'], tokenizer[f'r{result}'], tokenizer['<eos>']]
        tokens = (tokens + [tokenizer['<pad>']] * max_len)[:max_len]
        input_ids.append(tokens[:-1])
        labels.append(tokens[1:])
    return {'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels)}

def collate_reverse(batch, tokenizer, max_len=30):
    input_ids, labels = [], []
    for s, rev in batch:
        tokens = [tokenizer['<bos>']] + [tokenizer[f'v{v}'] for v in s]
        tokens += [tokenizer['<sep>']] + [tokenizer[f'v{v}'] for v in rev] + [tokenizer['<eos>']]
        tokens = (tokens + [tokenizer['<pad>']] * max_len)[:max_len]
        input_ids.append(tokens[:-1])
        labels.append(tokens[1:])
    return {'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels)}

def collate_modular(batch, tokenizer, max_len=40):
    input_ids, labels = [], []
    for expr, result in batch:
        tokens = [tokenizer['<bos>']] + [tokenizer[c] for c in expr if c in tokenizer]
        tokens += [tokenizer['<sep>'], tokenizer[f'r{result}'], tokenizer['<eos>']]
        tokens = (tokens + [tokenizer['<pad>']] * max_len)[:max_len]
        input_ids.append(tokens[:-1])
        labels.append(tokens[1:])
    return {'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels)}


# ============================================================================
# Model
# ============================================================================

def create_model(model_type, vocab_size, device, d_model=128, n_layers=4):
    config = ModelConfig(
        d_model=d_model, n_heads=4, n_layers=n_layers, mlp_ratio=4,
        vocab_size=vocab_size, max_sequence_length=64, embedding_size=vocab_size,
        weight_tying=True, include_bias=True, flash_attention=False,
        attention_dropout=0.0, residual_dropout=0.1, embedding_dropout=0.1,
        rope=True, init_device=device,
        use_ATF=(model_type in ['fanformer', 'both']), p_ratio=0.25,
        use_stack_memory=(model_type in ['stacktrans', 'both']),
        num_mem_heads=4, stack_slots=16, stack_dim=16,
    )
    return OLMo(config).to(device)


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, pad_id):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids).logits
        mask = (labels != pad_id).float()
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
        loss = (loss.view_as(labels) * mask).sum() / mask.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, sep_id, pad_id=None, is_seq=False):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            preds = model(input_ids).logits.argmax(-1)
            
            for i in range(input_ids.size(0)):
                sep_pos = (input_ids[i] == sep_id).nonzero()
                if len(sep_pos) > 0:
                    start = sep_pos[0].item()
                    if is_seq:
                        for j in range(start, labels.size(1)):
                            if labels[i, j] != pad_id:
                                correct += (preds[i, j] == labels[i, j]).item()
                                total += 1
                    else:
                        if start < labels.size(1):
                            correct += (preds[i, start] == labels[i, start]).item()
                            total += 1
    return correct / total if total > 0 else 0


# ============================================================================
# Task Configs
# ============================================================================

TASKS = {
    'parity': ('Parity Check (RE)', ParityCheckDataset, create_binary_tokenizer, collate_binary, False),
    'cycle': ('Cycle Navigation (RE)', CycleNavigationDataset, create_cycle_tokenizer, collate_cycle, False),
    'reverse': ('Reverse String (DCF)', ReverseStringDataset, create_reverse_tokenizer, collate_reverse, True),
    'modular': ('Modular Arithmetic (DCF)', ModularArithmeticDataset, create_modular_tokenizer, collate_modular, False),
}

MODEL_COLORS = {
    'transformer': '#1f77b4',
    'fanformer': '#ff7f0e', 
    'stacktrans': '#2ca02c',
    'both': '#d62728',
}

MODEL_MARKERS = {
    'transformer': 'o',
    'fanformer': 's',
    'stacktrans': '^',
    'both': 'D',
}


def run_task(task_name, model_types, epochs, batch_size, lr, device):
    name, DatasetClass, tok_fn, collate_fn, is_seq = TASKS[task_name]
    print(f"\n{'='*60}")
    print(f"任务: {name}")
    print(f"{'='*60}")
    
    tokenizer = tok_fn()
    vocab_size = len(tokenizer)
    pad_id, sep_id = tokenizer['<pad>'], tokenizer['<sep>']
    
    train_dataset = DatasetClass(num_samples=5000, seed=42)
    test_dataset = DatasetClass(num_samples=1000, seed=123)
    
    collate = lambda b: collate_fn(b, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    
    histories = {}
    
    for model_type in model_types:
        print(f"\n--- {model_type.upper()} ---")
        model = create_model(model_type, vocab_size, device)
        params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {params:,}")
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        train_losses, test_accs = [], []
        
        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device, pad_id)
            test_acc = evaluate(model, test_loader, device, sep_id, pad_id, is_seq)
            
            train_losses.append(train_loss)
            test_accs.append(test_acc)
            
            if epoch % 20 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | Acc: {test_acc:.4f}")
        
        histories[model_type] = {
            'train_losses': train_losses,
            'test_accs': test_accs,
            'params': params,
        }
        del model
    
    return name, histories


def plot_results(task_name, task_title, histories, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training Loss
    ax1 = axes[0]
    for model_type, hist in histories.items():
        epochs = range(1, len(hist['train_losses']) + 1)
        ax1.plot(epochs, hist['train_losses'], 
                color=MODEL_COLORS[model_type],
                marker=MODEL_MARKERS[model_type],
                markevery=max(1, len(epochs)//10),
                label=f"{model_type} ({hist['params']:,} params)",
                linewidth=2, markersize=6)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title(f'{task_title} - Training Loss', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test Accuracy
    ax2 = axes[1]
    for model_type, hist in histories.items():
        epochs = range(1, len(hist['test_accs']) + 1)
        ax2.plot(epochs, hist['test_accs'],
                color=MODEL_COLORS[model_type],
                marker=MODEL_MARKERS[model_type],
                markevery=max(1, len(epochs)//10),
                label=f"{model_type}",
                linewidth=2, markersize=6)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    ax2.set_title(f'{task_title} - Test Accuracy', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    filename = f"{output_dir}/{task_name}_training.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {filename}")
    plt.close()
    
    return filename


def plot_comparison(all_results, output_dir='plots'):
    """绘制所有任务的对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = list(all_results.keys())
    models = list(all_results[tasks[0]][1].keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(tasks))
    width = 0.2
    
    for i, model in enumerate(models):
        accs = [all_results[t][1][model]['test_accs'][-1] for t in tasks]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], accs, width, 
                      label=model, color=MODEL_COLORS[model])
        
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Final Test Accuracy', fontsize=12)
    ax.set_title('Model Comparison Across Formal Language Tasks', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([all_results[t][0] for t in tasks], rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    filename = f"{output_dir}/comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n对比图已保存: {filename}")
    plt.close()
    
    return filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='all', choices=['all', 'parity', 'cycle', 'reverse', 'modular'])
    parser.add_argument('--model', default='all', choices=['transformer', 'fanformer', 'stacktrans', 'both', 'all'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output_dir', default='plots')
    args = parser.parse_args()
    
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else \
                      ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Formal Language Tasks with Visualization")
    print(f"设备: {args.device}")
    
    model_types = ['transformer', 'fanformer', 'stacktrans', 'both'] if args.model == 'all' else [args.model]
    tasks = list(TASKS.keys()) if args.task == 'all' else [args.task]
    
    all_results = {}
    
    for task_name in tasks:
        task_title, histories = run_task(task_name, model_types, args.epochs, args.batch_size, args.lr, args.device)
        all_results[task_name] = (task_title, histories)
        plot_results(task_name, task_title, histories, args.output_dir)
    
    if len(tasks) > 1:
        plot_comparison(all_results, args.output_dir)
    
    # 汇总表格
    print(f"\n{'='*80}")
    print("最终结果汇总")
    print(f"{'='*80}")
    
    header = f"{'任务':<25}"
    for m in model_types:
        header += f" {m:<12}"
    print(header)
    print("-" * 80)
    
    for task_name in tasks:
        task_title, histories = all_results[task_name]
        row = f"{task_title:<25}"
        for m in model_types:
            row += f" {histories[m]['test_accs'][-1]:<12.4f}"
        print(row)
    
    print(f"\n图表保存在: {args.output_dir}/")


if __name__ == '__main__':
    main()
