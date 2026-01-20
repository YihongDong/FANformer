from __future__ import annotations

import os
import random
import re
from typing import Any

import torch
import torch.nn.functional as F  
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    AutoConfig
)
import math
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

from src.memory import MemoryModule
from src.utils import CfgNode
from src.utils import print0 as print0_origin
print0 = print0_origin if os.environ.get("DEBUG", "0") == "1" else lambda *args, **kwargs: None


class StackMemory(nn.Module):
    
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_mem_heads = config.num_mem_heads
        self.stack_slots = config.stack_slots
        self.head_dim = config.hidden_size // self.num_mem_heads
        
        
        self.action_head = nn.Linear(config.hidden_size, 3 * self.num_mem_heads)
        self.gate_proj = nn.Linear(self.head_dim, 1)
        self.res_weight = nn.Parameter(torch.ones(1))
        
        
        self.cache_size = getattr(config, "cache_size", 2048)
        self.register_buffer("k_cache", torch.zeros(self.cache_size, self.num_mem_heads, self.head_dim),persistent=False)
        self.register_buffer("action_cache", torch.zeros(self.cache_size, self.num_mem_heads, 3),persistent=False)
        self.cache_position = 0
        self.enable_cache = False

    def reset_cache(self):
        
        self.cache_position = 0

    def _vectorized_update(self, stack, mask, actions, k_values):
        
        
        
        
        
        
        batch_size, seq_len = actions.shape[:2]
        print0("before _vectorized_update stack:", stack.shape)
        print0("before _vectorized_update k_values:", k_values.shape)
        
        
        
        
        push_stack = torch.cat([
            k_values.unsqueeze(3),  
            stack[:, :, :, :-1]     
        ], dim=3)
        push_mask = torch.cat([
            torch.ones_like(mask[:, :, :, :1]),
            mask[:, :, :, :-1]
        ], dim=3)
        
        
        pop_stack = torch.cat([
            stack[:, :, :, 1:],
            torch.zeros_like(stack[:, :, :, :1])
        ], dim=3)
        pop_mask = torch.cat([
            mask[:, :, :, 1:],
            torch.zeros_like(mask[:, :, :, :1])
        ], dim=3)
        
        
        action_weights = actions.unsqueeze(-1).unsqueeze(-1)  
        stacks = torch.stack([push_stack, pop_stack, stack], dim=3)
        masks = torch.stack([push_mask, pop_mask, mask], dim=3)
        
        print0("action_weights:", action_weights.shape)
        print0("stacks:", stacks.shape)
        print0("masks:", masks.shape)
        new_stack = (stacks * action_weights).sum(dim=3)
        new_mask = (masks * action_weights.squeeze(-1)).sum(dim=3)
        print0("new_stack:", new_stack.shape)
        
        return new_stack, new_mask

    def forward(self, hidden_states, stack, mask):
        
        batch_size, seq_len, _ = hidden_states.shape
        
        
        action_logits = self.action_head(hidden_states) / math.sqrt(self.head_dim)
        actions = F.softmax(
            action_logits.view(batch_size, seq_len, self.num_mem_heads, 3), 
            dim=-1
        )  
        
        
        k_values = hidden_states.view(batch_size, seq_len, self.num_mem_heads, self.head_dim)
        
        
        new_stack, new_mask = self._vectorized_update(stack, mask, actions, k_values)
        
        
        
        gate_scores = self.gate_proj(new_stack).squeeze(-1)  
        gate_weights = F.softmax(gate_scores + (1 - new_mask) * -1e9, dim=-1)
        
        
        memory_output = (new_stack * gate_weights.unsqueeze(-1)).sum(dim=3)
        memory_output = memory_output.view(batch_size, seq_len, -1)
        
        
        output = memory_output * self.res_weight + hidden_states
        
        
        if self.training and self.enable_cache:
            self._update_cache(k_values.detach(), actions.detach())
        print0("new_stack:", new_stack.shape)
        print0("output:", output.shape)
        
        return output, new_stack, new_mask

    def _update_cache(self, k_values, actions):
        
        seq_len = k_values.shape[1]
        if self.cache_position + seq_len <= self.cache_size:
            self.k_cache[self.cache_position:self.cache_position+seq_len] = k_values[0]
            self.action_cache[self.cache_position:self.cache_position+seq_len] = actions[0]
            self.cache_position += seq_len
        else:
            self.reset_cache()

    def step(self, hidden_state, stack, mask):
        
        if not self.enable_cache:
            return self.forward(hidden_state.unsqueeze(1), stack, mask)
            
        
        if self.cache_position > 0:
            cached_k = self.k_cache[:self.cache_position]
            cached_actions = self.action_cache[:self.cache_position]
            
            
            k_values = torch.cat([cached_k.unsqueeze(0), hidden_state], dim=1)
            actions = torch.cat([cached_actions.unsqueeze(0), 
                               self.action_head(hidden_state).softmax(dim=-1)], dim=1)
        else:
            k_values = hidden_state
            actions = self.action_head(hidden_state).softmax(dim=-1)
        
        
        new_stack, new_mask = self._vectorized_update(
            stack.unsqueeze(1), 
            mask.unsqueeze(1), 
            actions.unsqueeze(0), 
            k_values.unsqueeze(0)
        )
        
        
        gate_scores = self.gate_proj(new_stack).squeeze(-1)
        gate_weights = F.softmax(gate_scores + (1 - new_mask) * -1e9, dim=-1)
        memory_output = (new_stack * gate_weights.unsqueeze(-1)).sum(dim=3)
        
        
        self._update_cache(k_values, actions)
        
        return (
            memory_output.squeeze(0) * self.res_weight + hidden_state,
            new_stack.squeeze(0),
            new_mask.squeeze(0)
        )



































class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    

    def __init__(self, config: LlamaConfig, layer_idx: int, use_memory: bool) -> None:
        super().__init__(config, layer_idx)
        self.use_memory = config.use_memory
        if self.use_memory:
            self.mem_stack = StackMemory(config)
        else:
            self.mem_stack = nn.Linear(config.hidden_size, config.hidden_size)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor = None,
        memory: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        position_embeddings: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        if self.use_memory and memory is not None:
            print0("=========> Using memory module")
            memory = memory.to(hidden_states.dtype)
            hidden_states, memory, memory_mask = self.mem_stack(
                hidden_states, memory, memory_mask
            )
        hidden_states = self.input_layernorm(hidden_states)

        
        
        
        
        
        
        
        
        
        attn_output, _, _ = self.self_attn(
            hidden_states,
            attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, memory, memory_mask


class LlamaMem(LlamaForCausalLM):
    

    @staticmethod
    def get_default_config():
        C = CfgNode()
        C.model_type = "llama"
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        C.vocab_size = None
        C.block_size = None
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        C.stack_slots = 16
        return C

    def __init__(self, config: LlamaConfig, tokenizer: AutoTokenizer) -> None:
        super().__init__(config)
        self.use_memory = config.use_memory
        self.tokenizer = tokenizer

        if self.use_memory:
            
            
            
            
            
            
            self.memory = torch.stack(
                [torch.zeros(config.seq_len, config.num_mem_heads, config.stack_slots, config.hidden_size // config.num_mem_heads, requires_grad=False) for _ in range(config.batch_size)]
            )
            self.memory_mask = torch.stack(
                [torch.zeros(config.seq_len, config.num_mem_heads,config.stack_slots, requires_grad=False) for _ in range(config.batch_size)]
            )
            
            self.register_buffer("memory_bank", self.memory, persistent=False)
            self.register_buffer("memory_mask_bank", self.memory_mask, persistent=False)
            print0("=========> Added memory module")
        else:
            print0("=========> No memory module")
            self.memory = None
            self.memory_mask = None

        
        self.model.layers = nn.ModuleList(
            [CustomLlamaDecoderLayer(config, layer_idx=i, use_memory=True) for i in range(len(self.model.layers))]
        )

    @classmethod
    def from_config(
        cls,
        config: LlamaConfig,
        tokenizer: AutoTokenizer,
    ) -> nn.Module:
        
        custom_model = cls(config, tokenizer)

        return custom_model

    @classmethod
    def from_ckpt(cls, pretrained_ckpt_path: str, config: LlamaConfig, 
                tokenizer: AutoTokenizer, rank: int, 
                load_memory: bool=True, resume_training: bool=False) -> nn.Module:
        
        try:
            
            snapshot_data = torch.load(pretrained_ckpt_path, map_location='cpu')

            custom_model = cls(config, tokenizer)

            
            model_state = snapshot_data['model_state_dict']

            
            custom_model.load_state_dict(model_state, strict=False)
            missing_keys, unexpected_keys = custom_model.load_state_dict(model_state, strict=False)
            if missing_keys or unexpected_keys:
                print0(f"Missing keys: {missing_keys}")
                print0(f"Unexpected keys: {unexpected_keys}")

            
            custom_model = custom_model.to(rank)
            if load_memory:
                custom_model.memory = custom_model.memory_bank.detach().clone()
                custom_model.memory_mask = custom_model.memory_mask.detach().clone()
                print0("===>Loaded pre-trained memory")
            else:
                
                
                
                
                custom_model.memory = torch.stack(
                    [torch.zeros(config.seq_len, config.num_mem_heads, config.stack_slots, config.hidden_size // config.num_mem_heads, requires_grad=False) for _ in range(config.batch_size)]
                )
                custom_model.memory_mask = torch.stack(
                    [torch.zeros(config.seq_len, config.num_mem_heads,config.stack_slots, requires_grad=False) for _ in range(config.batch_size)]
                )
                
                custom_model.register_buffer("memory_bank", custom_model.memory, persistent=False)
                custom_model.register_buffer("memory_mask_bank", custom_model.memory_mask, persistent=False)
                print0("===>Initialized fresh memory")

            
            try:
                random.setstate(snapshot_data['random_state'])

                
                torch_state = snapshot_data['torch_random_state']
                if isinstance(torch_state, torch.Tensor):
                    torch_state = torch_state.cpu().to(torch.uint8)
                else:
                    torch_state = torch.ByteTensor(torch_state)
                torch.set_rng_state(torch_state)

                
                cuda_states = snapshot_data['cuda_random_state']
                for i, state in enumerate(cuda_states):
                    if isinstance(state, torch.Tensor):
                        state = state.cpu().to(torch.uint8)
                    else:
                        state = torch.ByteTensor(state)
                    torch.cuda.set_rng_state(state, device=i)
                        
            except Exception as e:
                print0(f"Warning: Failed to restore random states: {str(e)}")
                print0("Continuing without restoring random states...")

            print0(f"Loaded checkpoint from iteration {snapshot_data['iteration']}")
            print0(f"Checkpoint learning rate: {snapshot_data['lr']}")

            if resume_training:
                return (custom_model, 
                        snapshot_data['optimizer_state_dict'],
                        snapshot_data['iteration'])

            return custom_model

        except Exception as e:
            print0(f"Error loading checkpoint: {e}")
            raise

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        iter_num: int = 0,
        num_logits_to_keep: int = 0,
    ) -> tuple[float, torch.Tensor]:
        
        device = input_ids.device
        b, t = input_ids.size()

        
        if self.use_memory:
            if self.memory is None or self.memory.device != device:
                self.memory = self.memory.to(device)
            memory = self.memory.detach()[:b, :t]
            if self.memory_mask is None or self.memory_mask.device != device:
                self.memory_mask = self.memory_mask.to(device)
            memory_mask = self.memory_mask.detach()[:b, :t]
            print0("memory:",memory.shape)
            print0("memory_mask:",memory_mask.shape)
        else:
            memory = None
            memory_mask = None

        
        if position_ids is None:
            position_ids = torch.arange(t, device=device).unsqueeze(0).repeat(b, 1)

        inputs_embeds = self.model.embed_tokens(input_ids)

        
        past_key_values = DynamicCache()
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        
        
        
        
        
        
        causal_mask = self.model._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            False,
        )
        
        
        
        
        if causal_mask.shape[-1] == causal_mask.shape[-2] + 1:
            
            causal_mask = causal_mask[:, :, :, :-1]
            
        
        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.model.layers:
            
            hidden_states, memory, memory_mask = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                memory=memory,
                memory_mask=memory_mask,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.tokenizer.pad_token_id,
            )
        
        
        
        
        
        
        
        dummy_loss = 0.0
        
        

        if loss is not None:
            loss += dummy_loss

        return logits, loss, memory

    def configure_optimizers(self, train_config):
        
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (
            torch.nn.LayerNorm,
            torch.nn.Embedding,
            LlamaRMSNorm,
        )
        pattern1 = re.compile(r"^transformer\.h\.[0-9]+\.mem_attn\.memory_module\.input_gate_projector\.w$")
        pattern2 = re.compile(r"^model\.layers\.\d+\.mem_attn\.memory_module\.input_gate_projector\.w$")

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  
                if "lm_head" in fpn:
                    
                    
                    no_decay.add(fpn)
                    continue
                if "mem_stack" in fpn:
                    decay.add(fpn)
                    continue
                elif pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pattern1.match(fpn) or pattern2.match(fpn):
                    no_decay.add(fpn)

        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {str(inter_params)} in both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"Parameters {str(param_dict.keys() - union_params)} not in either set!"

        
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer


class CustomLlamaConfig(LlamaConfig):
    

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.use_memory = kwargs.get("use_memory", False)
        self.num_mem_heads = kwargs.get("num_mem_heads", 4)
        self.log_freq = kwargs.get("log_freq", 100)
        self.batch_size = kwargs.get("batch_size", 1)
        self.stack_slots = kwargs.get("stack_slots", 8)
        self.seq_len = kwargs.get("seq_len", 2048)



if __name__ == "__main__":
    
    import os
    os.environ["RANK"] = "0"
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama-1.1B-Chat-v1.0")
    config = AutoConfig.from_pretrained("TinyLlama-1.1B-Chat-v1.0")
    config.use_memory = True
    
    config.num_mem_heads = 4
    config.log_freq = 100
    config.batch_size = 3
    config.stack_slots = 8
    config.seq_len = 3000
    
    
    
    
    
    
    
    model = LlamaMem(config, tokenizer)
    model.cpu()
    
    input_ids = torch.randint(0, 32000, (2, 2048))  
    import time
    start = time.time()
    logits, loss, memory = model(input_ids)
    end = time.time()
    print("time:", end-start)
    print(logits.shape)  