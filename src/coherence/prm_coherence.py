"""
Process Reward Model (PRM) baseline for evaluating CoT chains.
"""

from typing import List, Dict, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class PRMCoherenceMetric(nn.Module):
    """
    Compute internal coherence of a Chain-of-Thought using a Process Reward Model.
    
    Substitutes all classical/NLI metrics with a step-by-step reward assigned 
    by a specialized reasoning evaluator (e.g., ActPRM-X).
    """

    def __init__(
        self, 
        model_name: str = "sail/ActPRM-X",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(
                model_name,
                torch_dtype="auto",
                trust_remote_code=True,
            )
            .eval()
            .to(self.device)
        )
        
        # Cache the token ID for the step separator
        self.step_sep_id = self.tokenizer.encode("<extra_0>")[0]

    def _make_step_rewards(self, logits: torch.Tensor, token_masks: torch.Tensor) -> List[float]:
        """Extracts the scalar reward for each step from the model logits."""
        res = []
        # logits shape is typically [batch_size, seq_len, num_classes] or similar.
        # ActPRM-X provides specific step rewards at the `<extra_0>` tokens.
        for j in range(logits.size(1)):
            step_logits = logits[:, j, token_masks[j]]
            step_logits = step_logits.mean(dim=0)
            res.append(step_logits.tolist())
        return res

    def forward(
        self, 
        query: str, 
        steps: List[str], 
        system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate a single reasoning chain.
        """
        if not steps:
            return {'overall': torch.tensor(0.0), 'step_rewards': []}

        # Format input exactly as required by ActPRM-X
        response_str = "<extra_0>".join(steps) + "<extra_0>"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response_str},
        ]
        
        conversation_str = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )

        input_ids = self.tokenizer.encode(
            conversation_str,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.inference(input_ids=input_ids)

        token_masks = (input_ids == self.step_sep_id)
        
        # Outputs[0] contains the relevant logits based on the ActPRM-X implementation
        step_rewards = self._make_step_rewards(outputs[0], token_masks)
        
        # Convert list of lists to a flat 1D tensor
        # PRMs usually output a single float per step representing correctness/coherence
        step_rewards_tensor = torch.tensor([r[0] if isinstance(r, list) else r for r in step_rewards])
        
        # In PRMs, the minimum step reward (the weakest link) or the average is used for overall chain quality
        overall_score = step_rewards_tensor.mean()
        min_step_score = step_rewards_tensor.min()

        return {
            'overall': overall_score,
            'min_step_reward': min_step_score, # Highly indicative of a hallucination/logic break
            'step_rewards': step_rewards_tensor
        }

    def compute_for_batch(
        self, 
        batch_queries: List[str],
        batch_steps: List[List[str]],
        system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."
    ) -> List[Dict[str, torch.Tensor]]:
        
        results = []
        for i, query in enumerate(batch_queries):
            scores = self.forward(query=query, steps=batch_steps[i], system_prompt=system_prompt)
            results.append(scores)
            
        return results