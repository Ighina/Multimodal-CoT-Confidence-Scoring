"""
Internal logical coherence metrics for CoT chains using NLI.
"""

from typing import List, Optional, Dict
import torch
import torch.nn as nn
from transformers import pipeline

class NLICoherenceMetric(nn.Module):
    """
    Compute internal coherence of a Chain-of-Thought using Natural Language Inference.
    
    Measures:
    1. Cumulative Step Entailment: Does Step N+1 logically follow from [Step 1 ... Step N]?
    2. Final Goal Entailment: Does the final answer logically follow from the entire chain?
    """

    def __init__(
        self, 
        model_name: str = "cross-encoder/nli-deberta-v3-large",
        device: int = -1  # Set to 0 for CUDA
    ):
        super().__init__()
        # Load an NLI cross-encoder. 
        # Returns scores for: Contradiction, Entailment, Neutral (labels vary by model)
        self.nli_pipe = pipeline("text-classification", model=model_name, device=device, top_k=None)
        
    def _get_entailment_prob(self, premise: str, hypothesis: str) -> float:
        """Helper to extract the entailment probability from the pipeline."""
        text_pair = {"text": premise, "text_pair": hypothesis}
        results = self.nli_pipe(text_pair)
        
        # Depending on the model, the label might be 'entailment' or 'LABEL_1', etc.
        # cross-encoder/nli-deberta-v3-large uses 'entailment'
        for score_dict in results:
            if score_dict['label'].lower() == 'entailment':
                return score_dict['score']
        return 0.0

    def compute_cumulative_step_nli(self, steps: List[str]) -> torch.Tensor:
        """
        Compute average NLI probability between all previous steps and the next one.
        """
        if len(steps) < 2:
            return torch.tensor(1.0)

        entailment_scores = []
        
        for i in range(1, len(steps)):
            # Premise is all steps up to i
            premise = " ".join(steps[:i])
            # Hypothesis is step i
            hypothesis = steps[i]
            
            score = self._get_entailment_prob(premise, hypothesis)
            entailment_scores.append(score)
            
        return torch.tensor(entailment_scores).mean()

    def compute_goal_nli(self, steps: List[str], final_answer: str) -> torch.Tensor:
        """
        Compute NLI probability between all concatenated steps and the final answer.
        """
        premise = " ".join(steps)
        hypothesis = final_answer
        
        score = self._get_entailment_prob(premise, hypothesis)
        return torch.tensor(score)

    def forward(
        self, 
        steps: List[str], 
        final_answer: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        
        cumulative_step_score = self.compute_cumulative_step_nli(steps)
        
        if final_answer:
            goal_score = self.compute_goal_nli(steps, final_answer)
        else:
            # If no explicit answer is provided, check if the last step entails from all previous
            goal_score = self.compute_goal_nli(steps[:-1], steps[-1]) if len(steps) > 1 else torch.tensor(1.0)
            
        overall_score = (cumulative_step_score + goal_score) / 2.0

        return {
            'overall': overall_score,
            'cumulative_step_nli': cumulative_step_score,
            'goal_nli': goal_score
        }

    def compute_for_batch(
        self, 
        batch_steps: List[List[str]], 
        batch_final_answers: Optional[List[str]] = None
    ) -> List[Dict[str, torch.Tensor]]:
        
        results = []
        for i, steps in enumerate(batch_steps):
            answer = batch_final_answers[i] if batch_final_answers else None
            scores = self.forward(steps=steps, final_answer=answer)
            results.append(scores)
            
        return results