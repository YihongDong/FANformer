"""
Stack Memory module for OLMo, adapted from v4 stack transformer implementation.
Provides stack-based external memory with push, pop, and noop operations.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class StackMemory(nn.Module):
    """
    Stack Memory module that implements vectorized stack operations for transformer models.
    
    This module provides external stack-based memory with three operations:
    - Push: Add new element to top of stack
    - Pop: Remove element from top of stack  
    - Noop: Keep stack unchanged
    
    The operations are learned through attention-like mechanisms and can be applied
    in a differentiable, vectorized manner across batches and sequences.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_mem_heads = config.num_mem_heads
        self.stack_slots = config.stack_slots
        
        # Calculate memory dimension using centralized property
        self.head_dim = config.effective_stack_memory_dim
        self.stack_dim = config.effective_stack_dim
        
        # Check if compression is enabled
        self.use_compression = (self.stack_dim != self.head_dim)
        
        # Action prediction head: outputs logits for push/pop/noop for each memory head
        self.action_head = nn.Linear(
            config.d_model, 
            3 * self.num_mem_heads,
            bias=config.include_bias,
            device=config.init_device
        )
        
        # Gate projection for attention over stack slots (works with compressed dimension)
        self.gate_proj = nn.Linear(
            self.stack_dim,  # Changed from self.head_dim to work with compressed stack
            1,
            bias=config.include_bias,
            device=config.init_device
        )
        
        # Compression layers (if compression is enabled)
        if self.use_compression:
            self.down_proj = nn.Linear(
                self.head_dim, 
                self.stack_dim,
                bias=config.include_bias,
                device=config.init_device
            )
            self.up_proj = nn.Linear(
                self.stack_dim,
                self.head_dim, 
                bias=config.include_bias,
                device=config.init_device
            )
        else:
            self.down_proj = None
            self.up_proj = None
        
        # Residual connection weight
        self.res_weight = nn.Parameter(
            torch.ones(1, device=config.init_device)
        )
        
        # Cache configuration (cache compressed values for efficiency)
        self.cache_size = config.memory_cache_size
        self.register_buffer(
            "k_cache", 
            torch.zeros(self.cache_size, self.num_mem_heads, self.stack_dim),  # Use compressed dimension
            persistent=False
        )
        self.register_buffer(
            "action_cache", 
            torch.zeros(self.cache_size, self.num_mem_heads, 3),
            persistent=False
        )
        self.cache_position = 0
        self.enable_cache = False

    def reset_parameters(self):
        """Initialize parameters following OLMo conventions."""
        # Initialize action head
        nn.init.normal_(self.action_head.weight, std=0.02)
        if self.action_head.bias is not None:
            nn.init.zeros_(self.action_head.bias)
            
        # Initialize gate projection  
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        if self.gate_proj.bias is not None:
            nn.init.zeros_(self.gate_proj.bias)
            
        # Initialize compression layers if present
        if self.down_proj is not None:
            nn.init.normal_(self.down_proj.weight, std=0.02)
            if self.down_proj.bias is not None:
                nn.init.zeros_(self.down_proj.bias)
        
        if self.up_proj is not None:
            nn.init.normal_(self.up_proj.weight, std=0.02)
            if self.up_proj.bias is not None:
                nn.init.zeros_(self.up_proj.bias)
            
        # Initialize residual weight
        nn.init.ones_(self.res_weight)

    def reset_cache(self):
        """Reset the cache position to start fresh."""
        self.cache_position = 0

    def _vectorized_update(
        self, 
        stack: torch.Tensor, 
        mask: torch.Tensor, 
        actions: torch.Tensor, 
        k_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized stack update operations.
        
        Args:
            stack: Current stack state [batch, seq_len, num_heads, stack_slots, head_dim]
            mask: Current stack mask [batch, seq_len, num_heads, stack_slots]
            actions: Action probabilities [batch, seq_len, num_heads, 3] (push/pop/noop)
            k_values: New values to potentially push [batch, seq_len, num_heads, head_dim]
            
        Returns:
            new_stack: Updated stack state
            new_mask: Updated mask state
        """
        # Push operation: add new value to top, shift others down
        push_stack = torch.cat([
            k_values.unsqueeze(3),  # New value at position 0
            stack[:, :, :, :-1]     # Shift existing values right
        ], dim=3)
        push_mask = torch.cat([
            torch.ones_like(mask[:, :, :, :1]),  # New slot is occupied
            mask[:, :, :, :-1]                   # Shift mask right
        ], dim=3)
        
        # Pop operation: remove top value, shift others up  
        pop_stack = torch.cat([
            stack[:, :, :, 1:],                      # Shift values left
            torch.zeros_like(stack[:, :, :, :1])     # Zero at the end
        ], dim=3)
        pop_mask = torch.cat([
            mask[:, :, :, 1:],                       # Shift mask left
            torch.zeros_like(mask[:, :, :, :1])      # Empty slot at end
        ], dim=3)
        
        # Combine operations using action weights
        action_weights = actions.unsqueeze(-1).unsqueeze(-1)  # [batch, seq, heads, 3, 1, 1]
        stacks = torch.stack([push_stack, pop_stack, stack], dim=3)  # [batch, seq, heads, 3, slots, dim]
        masks = torch.stack([push_mask, pop_mask, mask], dim=3)       # [batch, seq, heads, 3, slots]
        
        new_stack = (stacks * action_weights).sum(dim=3)
        new_mask = (masks * action_weights.squeeze(-1)).sum(dim=3)
        
        return new_stack, new_mask

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        memory_stack: torch.Tensor, 
        memory_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of stack memory.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, d_model]
            memory_stack: Current stack state [batch, seq_len, num_heads, stack_slots, head_dim]  
            memory_mask: Current stack mask [batch, seq_len, num_heads, stack_slots]
            
        Returns:
            output: Enhanced hidden states with memory integration
            new_stack: Updated stack state
            new_mask: Updated mask state
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Predict actions for each memory head (push/pop/noop)
        action_logits = self.action_head(hidden_states) / math.sqrt(self.head_dim)
        actions = F.softmax(
            action_logits.view(batch_size, seq_len, self.num_mem_heads, 3), 
            dim=-1
        )
        
        # Prepare key values for potential pushing
        k_values = hidden_states.view(batch_size, seq_len, self.num_mem_heads, self.head_dim)
        
        # Apply compression if enabled
        if self.use_compression:
            k_values = self.down_proj(k_values)  # Compress from head_dim to stack_dim
        
        # Update stack with vectorized operations
        new_stack, new_mask = self._vectorized_update(memory_stack, memory_mask, actions, k_values)
        
        # Compute attention over stack slots
        gate_scores = self.gate_proj(new_stack).squeeze(-1)  # [batch, seq, heads, slots]
        gate_weights = F.softmax(
            gate_scores + (1 - new_mask) * -1e9,  # Mask out empty slots
            dim=-1
        )
        
        # Aggregate memory output
        memory_output = (new_stack * gate_weights.unsqueeze(-1)).sum(dim=3)
        
        # Apply decompression if enabled
        if self.use_compression:
            memory_output = self.up_proj(memory_output)  # Decompress from stack_dim to head_dim
            
        memory_output = memory_output.view(batch_size, seq_len, -1)
        
        # Residual connection
        output = memory_output * self.res_weight + hidden_states
        
        # Update cache if enabled during training
        if self.training and self.enable_cache:
            self._update_cache(k_values.detach(), actions.detach())
        
        return output, new_stack, new_mask

    def _update_cache(self, k_values: torch.Tensor, actions: torch.Tensor):
        """Update the internal cache with new values and actions."""
        seq_len = k_values.shape[1]
        if self.cache_position + seq_len <= self.cache_size:
            self.k_cache[self.cache_position:self.cache_position+seq_len] = k_values[0]
            self.action_cache[self.cache_position:self.cache_position+seq_len] = actions[0]
            self.cache_position += seq_len
        else:
            self.reset_cache()

    def step(
        self, 
        hidden_state: torch.Tensor, 
        memory_stack: torch.Tensor, 
        memory_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single step inference with cache support.
        
        Args:
            hidden_state: Single step input [batch, d_model]
            memory_stack: Current stack state [batch, num_heads, stack_slots, head_dim]
            memory_mask: Current stack mask [batch, num_heads, stack_slots]
            
        Returns:
            output: Enhanced hidden state [batch, d_model]
            new_stack: Updated stack state [batch, num_heads, stack_slots, head_dim]
            new_mask: Updated mask state [batch, num_heads, stack_slots]
        """
        if not self.enable_cache:
            # Add seq_len=1 dimension for forward method compatibility
            output, new_stack, new_mask = self.forward(
                hidden_state.unsqueeze(1),      # [batch, 1, d_model]
                memory_stack.unsqueeze(1),      # [batch, 1, num_heads, stack_slots, head_dim]
                memory_mask.unsqueeze(1)        # [batch, 1, num_heads, stack_slots]
            )
            # Remove seq_len=1 dimension from outputs
            return (
                output.squeeze(1),              # [batch, d_model]
                new_stack.squeeze(1),           # [batch, num_heads, stack_slots, head_dim]
                new_mask.squeeze(1)             # [batch, num_heads, stack_slots]
            )
            
        # Get batch size and reshape hidden_state for head-wise processing
        batch_size = hidden_state.shape[0]
        
        # Predict actions for the current step
        action_logits = self.action_head(hidden_state) / math.sqrt(self.head_dim)
        current_actions = F.softmax(
            action_logits.view(batch_size, self.num_mem_heads, 3), 
            dim=-1
        )  # [batch, num_heads, 3]
        
        # Prepare key values for potential pushing
        current_k_values = hidden_state.view(batch_size, self.num_mem_heads, self.head_dim)
        
        # Apply compression if enabled
        if self.use_compression:
            current_k_values = self.down_proj(current_k_values)  # Compress from head_dim to stack_dim
        
        # Use cached values if available
        if self.cache_position > 0:
            cached_k = self.k_cache[:self.cache_position]           # [cache_pos, num_heads, stack_dim]
            cached_actions = self.action_cache[:self.cache_position] # [cache_pos, num_heads, 3]
            
            # Concatenate cached and current values with proper dimensions
            k_values = torch.cat([
                cached_k.unsqueeze(0).expand(batch_size, -1, -1, -1),  # [batch, cache_pos, num_heads, stack_dim]
                current_k_values.unsqueeze(1)                          # [batch, 1, num_heads, stack_dim]
            ], dim=1)  # [batch, cache_pos+1, num_heads, stack_dim]
            
            actions = torch.cat([
                cached_actions.unsqueeze(0).expand(batch_size, -1, -1, -1),  # [batch, cache_pos, num_heads, 3]
                current_actions.unsqueeze(1)                                 # [batch, 1, num_heads, 3]
            ], dim=1)  # [batch, cache_pos+1, num_heads, 3]
        else:
            k_values = current_k_values.unsqueeze(1)    # [batch, 1, num_heads, stack_dim]
            actions = current_actions.unsqueeze(1)      # [batch, 1, num_heads, 3]
        
        # Update stack with vectorized operations
        # Add seq_len dimension to memory tensors for _vectorized_update
        new_stack, new_mask = self._vectorized_update(
            memory_stack.unsqueeze(1),      # [batch, 1, num_heads, stack_slots, stack_dim]
            memory_mask.unsqueeze(1),       # [batch, 1, num_heads, stack_slots]
            actions[:, -1:],                # [batch, 1, num_heads, 3] - only last timestep
            k_values[:, -1:]                # [batch, 1, num_heads, stack_dim] - only last timestep
        )
        
        # Compute attention over stack slots
        gate_scores = self.gate_proj(new_stack).squeeze(-1)  # [batch, 1, num_heads, stack_slots]
        gate_weights = F.softmax(
            gate_scores + (1 - new_mask) * -1e9,  # Mask out empty slots
            dim=-1
        )
        
        # Aggregate memory output
        memory_output = (new_stack * gate_weights.unsqueeze(-1)).sum(dim=3)  # [batch, 1, num_heads, stack_dim]
        
        # Apply decompression if enabled
        if self.use_compression:
            memory_output = self.up_proj(memory_output)  # Decompress from stack_dim to head_dim
            
        memory_output = memory_output.view(batch_size, 1, -1)  # [batch, 1, d_model]
        
        # Apply residual connection
        output = memory_output.squeeze(1) * self.res_weight + hidden_state  # [batch, d_model]
        
        # Update cache with current values only
        self._update_cache(current_k_values.detach(), current_actions.detach())
        
        return (
            output,                    # [batch, d_model]
            new_stack.squeeze(1),      # [batch, num_heads, stack_slots, head_dim]
            new_mask.squeeze(1)        # [batch, num_heads, stack_slots]
        )