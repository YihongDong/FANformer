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
    
    
    def __init__(self, config, stack_slots=None, num_mem_heads=None, hidden_size=None):
        super().__init__()
        self.config = config
        self.stack_dim = config.stack_dim
        if num_mem_heads is None:
            self.num_mem_heads = config.num_mem_heads
        else:
            self.num_mem_heads = num_mem_heads
        if stack_slots is None:
            self.stack_slots = config.stack_slots
        else:
            self.stack_slots = stack_slots
        if hidden_size is None:
            self.hidden_size = config.hidden_size
        else:
            self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_mem_heads
        
        
        self.action_head = nn.Linear(self.hidden_size, 3 * self.num_mem_heads)
        self.gate_proj = nn.Linear(self.stack_dim, 1)
        self.res_weight = nn.Parameter(torch.ones(1))
        self.down_proj = nn.Linear(self.head_dim, self.stack_dim)
        self.up_proj = nn.Linear(self.stack_dim, self.head_dim)
        
        
        self.cache_size = getattr(config, "cache_size", 2048)
        
        
        self.cache_position = 0
        self.enable_cache = False


        
        
        self.monitor_data = {
            'push_weights': [],
            'pop_weights': [],
            'noop_weights': []
        }
        
        
        self.action_head.register_forward_hook(self._capture_actions)
    
    def _capture_actions(self, module, input, output):
        
        if not self.training:  
            return
            
        
        batch_size, seq_len = output.shape[:2]
        actions = output.view(batch_size, seq_len, self.num_mem_heads, 3)
        
        
        with torch.no_grad():
            mean_weights = torch.softmax(actions, dim=-1).mean(dim=(0,1,2))  
            
        
        self.monitor_data['push_weights'].append(mean_weights[0].item())
        self.monitor_data['pop_weights'].append(mean_weights[1].item())
        self.monitor_data['noop_weights'].append(mean_weights[2].item())

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
        k_values = self.down_proj(k_values)
        
        
        new_stack, new_mask = self._vectorized_update(stack, mask, actions, k_values)
        
        
        
        gate_scores = self.gate_proj(new_stack).squeeze(-1)  
        gate_weights = F.softmax(gate_scores + (1 - new_mask) * -1e9, dim=-1)
        
        
        memory_output = (new_stack * gate_weights.unsqueeze(-1)).sum(dim=3)
        memory_output = self.up_proj(memory_output)
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
            
            
            self.cache_position += seq_len
        else:
            self.reset_cache()

    def step(self, hidden_state, stack, mask):
        
        if not self.enable_cache:
            return self.forward(hidden_state.unsqueeze(1), stack, mask)
            
        
        if self.cache_position > 0:
            
            
            
            
            
            
                            
            pass
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


class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config, hidden_size, num_experts_per_tok=1, n_routed_experts=2, routed_scaling_factor=1, n_group=1, topk_group=1, norm_topk_prob=True):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob

        self.weight = nn.Parameter(torch.empty((2, self.hidden_size)))  
        self.register_buffer("e_score_correction_bias", torch.zeros(2))  
        
        self.monitor_data = {
            'router_0_scores': [],   
            'router_1_scores': [],    
        }

        
        self.register_forward_hook(self._capture_scores)

    def _capture_scores(self, module, input, output):
        
        if not self.training:  
            return

        
        if not hasattr(self, '_scores_for_choice'):
            return

        scores = self._scores_for_choice.detach()  

        
        with torch.no_grad():
            self.monitor_data['router_0_scores'].append(scores[:,0].mean().item())
            self.monitor_data['router_1_scores'].append(scores[:,-1].mean().item())

        
        del self._scores_for_choice
        del scores
        

    def forward(self, hidden_states):
        
        hidden_states = hidden_states.view(-1, self.hidden_size)
        
        
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        
        
        scores = router_logits.sigmoid()
        
        
        scores = scores + self.e_score_correction_bias.unsqueeze(0)
        
        if self.training:
            self._scores_for_choice = scores.detach()
        
        top_indices = scores.argmax(dim=-1)  
        top_scores = scores.gather(1, top_indices.unsqueeze(1)).squeeze(1)  
        
        return top_indices, top_scores


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
        self.use_outer_memory = config.use_outer_memory
        self.tokenizer = tokenizer

        self.outer_max_iter_num = config.outer_max_iter_num

        if self.use_memory:
            
            
            
            
            
            
            self.memory = torch.stack(
                
                [torch.zeros(config.seq_len, config.num_mem_heads, config.stack_slots, config.stack_dim, requires_grad=False) for _ in range(config.batch_size)]
            )
            self.memory_mask = torch.stack(
                [torch.zeros(config.seq_len, config.num_mem_heads,config.stack_slots, requires_grad=False) for _ in range(config.batch_size)]
            )
            
            
            
            
            print0("=========> Added memory module")
        else:
            print0("=========> No memory module")
            self.memory = None
            self.memory_mask = None
        if self.use_outer_memory:
            self.outer_stack = StackMemory(config, config.outer_max_iter_num, config.outer_num_mem_heads, config.hidden_size)
            self.outer_memory = torch.stack(
                
                [torch.zeros(config.seq_len, config.outer_num_mem_heads, config.outer_max_iter_num, config.stack_dim, requires_grad=False) for _ in range(config.batch_size)]
            )
            self.outer_memory_mask = torch.stack(
                [torch.zeros(config.seq_len, config.outer_num_mem_heads,config.outer_max_iter_num, requires_grad=False) for _ in range(config.batch_size)]
            )
            
            
            
            print0("=========> Added memory module")
        else:
            print0("=========> No memory module")
            self.outer_memory = None
            self.outer_memory_mask = None

        self.router = DeepseekV3TopkRouter(
            config=config,
            hidden_size=config.hidden_size,
            num_experts_per_tok=1,  
            n_routed_experts=2,     
            routed_scaling_factor=1.0,
            n_group=1,              
            topk_group=1,           
            norm_topk_prob=True
        )
        
        
        self.register_buffer('expert_map', torch.tensor([0, 1], dtype=torch.long))

        
        self.model.layers = nn.ModuleList(
            [CustomLlamaDecoderLayer(config, layer_idx=i, use_memory=True) for i in range(len(self.model.layers))]
        )

                
        self.monitor_data = {
            'exit_layer_num': [],   
            'max_exit_layer_num': []
        }

        
        self.register_forward_hook(self._capture_scores)

    def _capture_scores(self, module, input, output):
        
        if not self.training:  
            return

        
        if not hasattr(self, '_scores_for_exit'):
            return

        scores = self._scores_for_exit.detach()  

        
        with torch.no_grad():
            self.monitor_data['exit_layer_num'].append(scores.mean().item())
            self.monitor_data['max_exit_layer_num'].append(scores.max().item())

        
        del self._scores_for_exit
        del scores

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
                load_memory: bool=True, resume_training: bool=False, load_outer_memory: bool=True) -> nn.Module:
        
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
                
                
                print0("===>!!! No Load pre-trained memory")
            else:
                
                
                
                
                custom_model.memory = torch.stack(
                    [torch.zeros(config.seq_len, config.num_mem_heads, config.stack_slots, config.hidden_size // config.num_mem_heads, requires_grad=False) for _ in range(config.batch_size)]
                )
                custom_model.memory_mask = torch.stack(
                    [torch.zeros(config.seq_len, config.num_mem_heads,config.stack_slots, requires_grad=False) for _ in range(config.batch_size)]
                )
                
                
                
                print0("===>Initialized fresh memory")

            if load_outer_memory:
                
                
                print0("===>!!! No Load pre-trained outer memory")
            else:
                
                
                
                
                custom_model.outer_memory = torch.stack(
                    [torch.zeros(config.seq_len, config.outer_num_mem_heads, config.outer_max_iter_num, config.hidden_size // config.outer_num_mem_heads, requires_grad=False) for _ in range(config.batch_size)]
                )
                custom_model.outer_memory_mask = torch.stack(
                    [torch.zeros(config.seq_len, config.outer_num_mem_heads,config.outer_max_iter_num, requires_grad=False) for _ in range(config.batch_size)]
                )
                
                
                
                print0("===>Initialized fresh outer memory")

            
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

    @classmethod
    def from_hf_ckpt(cls, pretrained_model_name_or_path: str, config: LlamaConfig,
                    tokenizer: AutoTokenizer, rank: int,
                    load_memory: bool = False,
                    load_outer_memory: bool = False) -> nn.Module:
        
        try:
            
            base_model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                config=config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            
            custom_model = cls(config, tokenizer)
            
            
            state_dict = base_model.state_dict()
            
            
            
            
            
            
            missing_keys, unexpected_keys = custom_model.load_state_dict(
                state_dict, strict=False
            )
            print0(f"Load model...")
            
            
            
            
            
            
            print(f"Missing keys (non-memory): {missing_keys}")
            
            print(f"Unexpected keys (non-memory): {unexpected_keys}")

            
            custom_model = custom_model.to(rank)
            if load_memory:
                
                
                print0("===>!!! No Load pre-trained memory")
            else:
                
                
                
                
                custom_model.memory = torch.stack(
                    
                    [torch.zeros(config.seq_len, config.num_mem_heads, config.stack_slots, config.stack_dim, requires_grad=False) for _ in range(config.batch_size)]
                )
                custom_model.memory_mask = torch.stack(
                    [torch.zeros(config.seq_len, config.num_mem_heads,config.stack_slots, requires_grad=False) for _ in range(config.batch_size)]
                )
                
                
                
                print0("===>Initialized fresh memory")

            if load_outer_memory:
                
                
                print0("===>!!! No Load pre-trained outer memory")
            else:
                
                
                
                
                custom_model.outer_memory = torch.stack(
                    
                    [torch.zeros(config.seq_len, config.outer_num_mem_heads, config.outer_max_iter_num, config.stack_dim, requires_grad=False) for _ in range(config.batch_size)]
                )
                custom_model.outer_memory_mask = torch.stack(
                    [torch.zeros(config.seq_len, config.outer_num_mem_heads,config.outer_max_iter_num, requires_grad=False) for _ in range(config.batch_size)]
                )
                
                
                
                print0("===>Initialized fresh outer memory")

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

        if self.use_outer_memory:
            if self.outer_memory is None or self.outer_memory.device != device:
                self.outer_memory = self.outer_memory.to(device)
            outer_memory = self.outer_memory.detach()[:b, :t]
            if self.outer_memory_mask is None or self.outer_memory_mask.device != device:
                self.outer_memory_mask = self.outer_memory_mask.to(device)
            outer_memory_mask = self.outer_memory_mask.detach()[:b, :t]
            print0("outer_memory:",outer_memory.shape)
            print0("outer_memory_mask:",outer_memory_mask.shape)
        else:
            outer_memory = None
            outer_memory_mask = None

        
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
            attention_mask=attention_mask,
            input_tensor=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
            output_attentions = False,
        )
        
        
        
        
        if not (causal_mask is None) and causal_mask.shape[-1] == causal_mask.shape[-2] + 1:
            
            causal_mask = causal_mask[:, :, :, :-1]
            
        
        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        final_logits = None
        exit_mask = None
        
        if self.training:
            self._scores_for_exit = None

        for i in range(self.outer_max_iter_num):
            
            if self.use_outer_memory:
                
                
                hidden_states, outer_memory, outer_memory_mask = self.outer_stack(
                    hidden_states, outer_memory, outer_memory_mask
                )
            
            
            prev_hidden_states = hidden_states.clone()
            
            
            for decoder_layer in self.model.layers:
                hidden_states, memory, memory_mask = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    memory=memory,
                    memory_mask=memory_mask,
                    position_embeddings=position_embeddings,
                )
            
            
            current_hidden = self.model.norm(hidden_states)
            current_logits = self.lm_head(current_hidden[:, -num_logits_to_keep:, :]).float()
            batch_size, seq_len, _ = current_logits.shape
            
            
            topk_indices, topk_weights = self.router(current_hidden)
            selected_experts = self.expert_map[topk_indices]  
            
            
            if final_logits is None:
                final_logits = torch.zeros_like(current_logits)
                exit_mask = torch.zeros_like(selected_experts, dtype=torch.bool)
            
            
            
            
            exits_this_round = (selected_experts == 0) & ~exit_mask

            if self.training:
                this_round_num = exits_this_round.float().detach() * (i + 1)
                
                
                if self._scores_for_exit is None:
                    self._scores_for_exit = this_round_num.detach()
                else:
                    self._scores_for_exit = self._scores_for_exit + this_round_num.detach()
            
            
            final_logits = final_logits + current_logits * exits_this_round.view(batch_size, seq_len, 1) * topk_weights.view(batch_size, seq_len, 1)
            
            
            hidden_states = torch.where(exits_this_round.view(batch_size, seq_len, 1), prev_hidden_states, hidden_states)
            
            
            exit_mask = exit_mask | exits_this_round
            
            
            if exit_mask.all():
                break

        
        if not exit_mask.all():
            remaining = ~exit_mask
            if self.training:
                this_round_num = remaining.float().detach() * (self.outer_max_iter_num)
                self._scores_for_exit = self._scores_for_exit + this_round_num.detach()
            last_hidden = self.model.norm(hidden_states)
            last_logits = self.lm_head(last_hidden[:, -num_logits_to_keep:, :]).float()
            final_logits = final_logits + last_logits * remaining.view(batch_size, seq_len, 1) * topk_weights.view(batch_size, seq_len, 1)


        logits = final_logits

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.tokenizer.pad_token_id,
            )
        
        
        
        
        
        
        
        
        

        
        

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
                if "router" in fpn:
                    decay.add(fpn)
                    continue
                if "outer_stack" in fpn:
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

        
        weight_decay_params = [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict]
        no_weight_decay_params = [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict]
        optim_groups = [
            {
                "params": weight_decay_params,
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": no_weight_decay_params,
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer


class CustomLlamaConfig(LlamaConfig):
    

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.use_memory = kwargs.get("use_memory", True)
        self.num_mem_heads = kwargs.get("num_mem_heads", 4)
        self.log_freq = kwargs.get("log_freq", 100)
        self.batch_size = kwargs.get("batch_size", 1)
        self.stack_slots = kwargs.get("stack_slots", 8)
        self.seq_len = kwargs.get("seq_len", 2048)
        self.outer_max_iter_num = kwargs.get("outer_max_iter_num", 2)
        self.use_outer_memory = kwargs.get("use_outer_memory", True)
        self.outer_num_mem_heads = kwargs.get("outer_num_mem_heads", 4)
        self.stack_dim = kwargs.get("stack_dim", 4)


def unit_test_from_scratch():
    
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
    config.outer_max_iter_num = 5
    config.use_outer_memory = True
    config.outer_num_mem_heads = 4
    
    
    
    
    
    
    
    model = LlamaMem(config, tokenizer)
    model.cpu()
    
    input_ids = torch.randint(0, 32000, (2, 2048))  
    import time
    start = time.time()
    logits, loss, memory = model(input_ids)
    end = time.time()
    print("time:", end-start)
    print(logits.shape)  

def unit_test_cpt():
    import os
    os.environ["RANK"] = "0"
    inp_dir = "/mnt/bd/trainedmodel-ds1000/SmolLM2-360M"
    tokenizer = AutoTokenizer.from_pretrained(inp_dir)
    config = AutoConfig.from_pretrained(inp_dir)
    config.use_memory = True
    
    config.num_mem_heads = 4
    config.log_freq = 100
    config.batch_size = 3
    config.stack_slots = 8
    config.seq_len = 3000
    config.outer_max_iter_num = 5
    config.use_outer_memory = True
    config.outer_num_mem_heads = 4
    
    
    
    
    
    
    
    
    model= LlamaMem.from_hf_ckpt(
                    inp_dir,
                    config=config,
                    tokenizer=tokenizer,
                    rank=0,
                )
    
    model.cpu()
    
    input_ids = torch.randint(0, 32000, (2, 2048))  
    import time
    start = time.time()
    logits, loss, memory = model(input_ids)
    end = time.time()
    print("time:", end-start)
    print(logits.shape)  
    print("model parameter size:")
    print(sum(p.numel() for p in model.parameters()))
    print("billion parameter size:")
    print(sum(p.numel() for p in model.parameters()) / 1e9, "B")


if __name__ == "__main__":
    
    unit_test_cpt()