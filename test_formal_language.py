#!/usr/bin/env python3
"""
Formal Language Tasks - 测试 FANformer/StackTrans 在 Chomsky 层次结构任务上的表现

包含以下任务（参考论文 Table 1）:

1. Regular (RE) Tasks:
   - Parity Check: 检查字符串中 'b' 的数量是否为偶数
   - Even Pairs: 检查 ab/ba 对的数量是否为偶数
   - Cycle Navigation: 在 mod-5 循环上导航

2. Deterministic Context-Free (DCF) Tasks:
   - Reverse String: 反转字符串
   - Stack Manipulation: 模拟栈操作
   - Modular Arithmetic: 带括号的模运算表达式求值
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import argparse
from typing import List, Tuple

from olmo.config import ModelConfig
from olmo.model import OLMo


# ============================================================================
# Task 1: Parity Check (RE)
# 检查字符串中 'b' 的数量是否为偶数
# ============================================================================
class ParityCheckDataset(Dataset):
    """Parity Check: 输入 ab 字符串，输出 True/False"""
    
    def __init__(self, min_len=1, max_len=20, num_samples=5000, seed=42):
        random.seed(seed)
        self.samples = []
        
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            s = ''.join(random.choice('ab') for _ in range(length))
            label = s.count('b') % 2 == 0  # True if even number of 'b's
            self.samples.append((s, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# Task 2: Even Pairs (RE)
# 检查 ab/ba 对的数量是否为偶数
# ============================================================================
class EvenPairsDataset(Dataset):
    """Even Pairs: 检查 ab 或 ba 子串的数量是否为偶数"""
    
    def __init__(self, min_len=2, max_len=20, num_samples=5000, seed=42):
        random.seed(seed)
        self.samples = []
        
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            s = ''.join(random.choice('ab') for _ in range(length))
            # 统计 ab 和 ba 的数量
            count = sum(1 for i in range(len(s)-1) if s[i:i+2] in ['ab', 'ba'])
            label = count % 2 == 0
            self.samples.append((s, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# Task 3: Cycle Navigation (RE)
# 在 mod-5 循环上导航: 0=不动, 1=+1, 2=-1
# ============================================================================
class CycleNavigationDataset(Dataset):
    """Cycle Navigation: 从 0 开始，按指令移动，输出最终位置 mod 5"""
    
    def __init__(self, min_len=1, max_len=20, num_samples=5000, seed=42):
        random.seed(seed)
        self.samples = []
        
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            # 0=stay, 1=+1, 2=-1
            moves = [random.randint(0, 2) for _ in range(length)]
            position = 0
            for m in moves:
                if m == 1:
                    position = (position + 1) % 5
                elif m == 2:
                    position = (position - 1) % 5
            self.samples.append((moves, position))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# Task 4: Reverse String (DCF)
# 反转输入字符串
# ============================================================================
class ReverseStringDataset(Dataset):
    """Reverse String: 输入字符串，输出反转后的字符串"""
    
    def __init__(self, min_len=1, max_len=10, num_samples=5000, seed=42, vocab_size=5):
        random.seed(seed)
        self.samples = []
        self.vocab_size = vocab_size
        
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            s = [random.randint(0, vocab_size-1) for _ in range(length)]
            rev = s[::-1]
            self.samples.append((s, rev))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# Task 5: Balanced Parentheses (DCF)
# 判断括号是否匹配
# ============================================================================
class BalancedParenthesesDataset(Dataset):
    """Balanced Parentheses: 判断括号序列是否平衡"""
    
    def __init__(self, min_len=2, max_len=20, num_samples=5000, seed=42):
        random.seed(seed)
        self.samples = []
        
        for _ in range(num_samples):
            # 50% 生成平衡的，50% 生成随机的
            if random.random() < 0.5:
                s, label = self._generate_balanced(random.randint(min_len//2, max_len//2))
            else:
                length = random.randint(min_len, max_len)
                s = ''.join(random.choice('()') for _ in range(length))
                label = self._is_balanced(s)
            self.samples.append((s, label))
    
    def _generate_balanced(self, n):
        """生成 n 对平衡括号"""
        if n == 0:
            return '', True
        s = ['('] * n + [')'] * n
        random.shuffle(s)
        # 确保生成的是平衡的
        result = []
        count = 0
        for c in s:
            if c == '(':
                result.append(c)
                count += 1
            else:
                if count > 0:
                    result.append(c)
                    count -= 1
        while count > 0:
            result.append(')')
            count -= 1
        return ''.join(result), True
    
    def _is_balanced(self, s):
        count = 0
        for c in s:
            if c == '(':
                count += 1
            else:
                count -= 1
            if count < 0:
                return False
        return count == 0
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# Task 6: Modular Arithmetic with Parentheses (DCF)
# 计算带括号的模运算表达式
# ============================================================================
class ModularArithmeticDataset(Dataset):
    """Modular Arithmetic: 计算嵌套表达式 mod 5"""
    
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
        
        if op == '+':
            result = (left_val + right_val) % 5
        elif op == '-':
            result = (left_val - right_val) % 5
        else:  # *
            result = (left_val * right_val) % 5
        
        expr = f"({left_expr}{op}{right_expr})"
        return expr, result
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# Tokenizers and Collate Functions
# ============================================================================

def create_binary_classification_tokenizer():
    """用于二分类任务（Parity, EvenPairs, BalancedParentheses）"""
    return {
        '<pad>': 0, '<bos>': 1, '<eos>': 2, '<sep>': 3,
        'a': 4, 'b': 5, '(': 6, ')': 7,
        'T': 8, 'F': 9,  # True/False
    }


def create_cycle_tokenizer():
    """用于 Cycle Navigation"""
    return {
        '<pad>': 0, '<bos>': 1, '<eos>': 2, '<sep>': 3,
        '0': 4, '1': 5, '2': 6,  # moves
        'r0': 7, 'r1': 8, 'r2': 9, 'r3': 10, 'r4': 11,  # results
    }


def create_reverse_tokenizer(vocab_size=5):
    """用于 Reverse String"""
    tok = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<sep>': 3}
    for i in range(vocab_size):
        tok[f'v{i}'] = 4 + i
    return tok


def create_modular_tokenizer():
    """用于 Modular Arithmetic"""
    return {
        '<pad>': 0, '<bos>': 1, '<eos>': 2, '<sep>': 3,
        '0': 4, '1': 5, '2': 6, '3': 7, '4': 8,
        '+': 9, '-': 10, '*': 11, '(': 12, ')': 13,
        'r0': 14, 'r1': 15, 'r2': 16, 'r3': 17, 'r4': 18,
    }


def collate_binary_classification(batch, tokenizer, max_len=30):
    """二分类任务的 collate"""
    input_ids, labels = [], []
    
    for s, label in batch:
        tokens = [tokenizer['<bos>']]
        for c in s:
            if c in tokenizer:
                tokens.append(tokenizer[c])
        tokens.append(tokenizer['<sep>'])
        target = tokenizer['T'] if label else tokenizer['F']
        tokens.append(target)
        tokens.append(tokenizer['<eos>'])
        
        while len(tokens) < max_len:
            tokens.append(tokenizer['<pad>'])
        tokens = tokens[:max_len]
        
        input_ids.append(tokens[:-1])
        labels.append(tokens[1:])
    
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


def collate_cycle(batch, tokenizer, max_len=30):
    """Cycle Navigation 的 collate"""
    input_ids, labels = [], []
    
    for moves, result in batch:
        tokens = [tokenizer['<bos>']]
        for m in moves:
            tokens.append(tokenizer[str(m)])
        tokens.append(tokenizer['<sep>'])
        tokens.append(tokenizer[f'r{result}'])
        tokens.append(tokenizer['<eos>'])
        
        while len(tokens) < max_len:
            tokens.append(tokenizer['<pad>'])
        tokens = tokens[:max_len]
        
        input_ids.append(tokens[:-1])
        labels.append(tokens[1:])
    
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


def collate_reverse(batch, tokenizer, max_len=30):
    """Reverse String 的 collate"""
    input_ids, labels = [], []
    
    for s, rev in batch:
        tokens = [tokenizer['<bos>']]
        for v in s:
            tokens.append(tokenizer[f'v{v}'])
        tokens.append(tokenizer['<sep>'])
        for v in rev:
            tokens.append(tokenizer[f'v{v}'])
        tokens.append(tokenizer['<eos>'])
        
        while len(tokens) < max_len:
            tokens.append(tokenizer['<pad>'])
        tokens = tokens[:max_len]
        
        input_ids.append(tokens[:-1])
        labels.append(tokens[1:])
    
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


def collate_modular(batch, tokenizer, max_len=40):
    """Modular Arithmetic 的 collate"""
    input_ids, labels = [], []
    
    for expr, result in batch:
        tokens = [tokenizer['<bos>']]
        for c in expr:
            if c in tokenizer:
                tokens.append(tokenizer[c])
        tokens.append(tokenizer['<sep>'])
        tokens.append(tokenizer[f'r{result}'])
        tokens.append(tokenizer['<eos>'])
        
        while len(tokens) < max_len:
            tokens.append(tokenizer['<pad>'])
        tokens = tokens[:max_len]
        
        input_ids.append(tokens[:-1])
        labels.append(tokens[1:])
    
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


# ============================================================================
# Model Creation
# ============================================================================

def create_model(model_type, vocab_size, device, d_model=128, n_layers=4):
    config = ModelConfig(
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        mlp_ratio=4,
        vocab_size=vocab_size,
        max_sequence_length=64,
        embedding_size=vocab_size,
        weight_tying=True,
        include_bias=True,
        flash_attention=False,
        attention_dropout=0.0,
        residual_dropout=0.1,
        embedding_dropout=0.1,
        rope=True,
        init_device=device,
        
        use_ATF=(model_type in ['fanformer', 'both']),
        p_ratio=0.25,
        
        use_stack_memory=(model_type in ['stacktrans', 'both']),
        num_mem_heads=4,
        stack_slots=16,
        stack_dim=16,
    )
    return OLMo(config).to(device)


# ============================================================================
# Training and Evaluation
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
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none'
        )
        loss = (loss.view_as(labels) * mask).sum() / mask.sum()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate_classification(model, dataloader, device, tokenizer, sep_id):
    """评估分类任务（只看 <sep> 后面第一个位置的准确率）"""
    model.eval()
    correct = total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            preds = model(input_ids).logits.argmax(-1)
            
            # 找到 <sep> 的位置，检查下一个位置的预测
            sep_mask = (input_ids == sep_id)
            for i in range(input_ids.size(0)):
                sep_pos = sep_mask[i].nonzero()
                if len(sep_pos) > 0:
                    pos = sep_pos[0].item()
                    if pos < labels.size(1):
                        if preds[i, pos] == labels[i, pos]:
                            correct += 1
                        total += 1
    
    return correct / total if total > 0 else 0


def evaluate_sequence(model, dataloader, device, tokenizer, sep_id, pad_id):
    """评估序列生成任务（<sep> 之后所有非 pad 位置的准确率）"""
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
                    for j in range(start, labels.size(1)):
                        if labels[i, j] != pad_id:
                            if preds[i, j] == labels[i, j]:
                                correct += 1
                            total += 1
    
    return correct / total if total > 0 else 0


# ============================================================================
# Task Runners
# ============================================================================

TASKS = {
    'parity': {
        'name': 'Parity Check (RE)',
        'dataset_class': ParityCheckDataset,
        'tokenizer_fn': create_binary_classification_tokenizer,
        'collate_fn': collate_binary_classification,
        'eval_fn': evaluate_classification,
        'is_classification': True,
    },
    'even_pairs': {
        'name': 'Even Pairs (RE)',
        'dataset_class': EvenPairsDataset,
        'tokenizer_fn': create_binary_classification_tokenizer,
        'collate_fn': collate_binary_classification,
        'eval_fn': evaluate_classification,
        'is_classification': True,
    },
    'cycle': {
        'name': 'Cycle Navigation (RE)',
        'dataset_class': CycleNavigationDataset,
        'tokenizer_fn': create_cycle_tokenizer,
        'collate_fn': collate_cycle,
        'eval_fn': evaluate_classification,
        'is_classification': True,
    },
    'reverse': {
        'name': 'Reverse String (DCF)',
        'dataset_class': ReverseStringDataset,
        'tokenizer_fn': lambda: create_reverse_tokenizer(5),
        'collate_fn': collate_reverse,
        'eval_fn': evaluate_sequence,
        'is_classification': False,
    },
    'balanced': {
        'name': 'Balanced Parentheses (DCF)',
        'dataset_class': BalancedParenthesesDataset,
        'tokenizer_fn': create_binary_classification_tokenizer,
        'collate_fn': collate_binary_classification,
        'eval_fn': evaluate_classification,
        'is_classification': True,
    },
    'modular': {
        'name': 'Modular Arithmetic (DCF)',
        'dataset_class': ModularArithmeticDataset,
        'tokenizer_fn': create_modular_tokenizer,
        'collate_fn': collate_modular,
        'eval_fn': evaluate_classification,
        'is_classification': True,
    },
}


def run_task(task_name, model_types, epochs, batch_size, lr, device):
    """运行单个任务"""
    task = TASKS[task_name]
    print(f"\n{'='*60}")
    print(f"任务: {task['name']}")
    print(f"{'='*60}")
    
    tokenizer = task['tokenizer_fn']()
    vocab_size = len(tokenizer)
    pad_id = tokenizer['<pad>']
    sep_id = tokenizer['<sep>']
    
    # 创建数据集
    train_dataset = task['dataset_class'](num_samples=5000, seed=42)
    test_dataset = task['dataset_class'](num_samples=1000, seed=123)
    
    collate = lambda b: task['collate_fn'](b, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    
    results = {}
    
    for model_type in model_types:
        print(f"\n--- {model_type.upper()} ---")
        
        model = create_model(model_type, vocab_size, device)
        params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {params:,}")
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        best_acc = 0
        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device, pad_id)
            
            if task['is_classification']:
                test_acc = task['eval_fn'](model, test_loader, device, tokenizer, sep_id)
            else:
                test_acc = task['eval_fn'](model, test_loader, device, tokenizer, sep_id, pad_id)
            
            best_acc = max(best_acc, test_acc)
            
            if epoch % 20 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | Acc: {test_acc:.4f}")
        
        results[model_type] = {'params': params, 'best_acc': best_acc, 'final_acc': test_acc}
        del model
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Formal Language Tasks')
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'parity', 'even_pairs', 'cycle', 'reverse', 'balanced', 'modular'])
    parser.add_argument('--model', type=str, default='all',
                        choices=['transformer', 'fanformer', 'stacktrans', 'both', 'all'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else \
                      ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Formal Language Tasks")
    print(f"设备: {args.device}")
    
    model_types = ['transformer', 'fanformer', 'stacktrans', 'both'] if args.model == 'all' else [args.model]
    tasks = list(TASKS.keys()) if args.task == 'all' else [args.task]
    
    all_results = {}
    
    for task_name in tasks:
        all_results[task_name] = run_task(
            task_name, model_types, args.epochs, args.batch_size, args.lr, args.device
        )
    
    # 汇总
    print(f"\n{'='*80}")
    print("最终结果汇总")
    print(f"{'='*80}")
    
    header = f"{'任务':<25}"
    for m in model_types:
        header += f" {m:<12}"
    print(header)
    print("-" * 80)
    
    for task_name, results in all_results.items():
        row = f"{TASKS[task_name]['name']:<25}"
        for m in model_types:
            if m in results:
                row += f" {results[m]['best_acc']:<12.4f}"
            else:
                row += f" {'N/A':<12}"
        print(row)
    
    print()


if __name__ == '__main__':
    main()
