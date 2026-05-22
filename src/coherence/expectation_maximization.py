import numpy as np
from typing import List
from sklearn.mixture import GaussianMixture

def compute_em_fused_embeddings_only(
    embedding_arrays: List[np.ndarray],
) -> np.ndarray:
    """
    Fuses multiple orthogonal embedding scores using Expectation-Maximization
    to estimate the latent probability of correctness.
    
    Args:
        embedding_arrays: List of numpy arrays, all of shape (n_examples, n_chains).
                          e.g., [cross_modal_score, internal_smoothness, nli_score]
                          
    Returns:
        EM-fused confidence scores of the same shape.
    """
    original_shape = embedding_arrays[0].shape
    
    # Flatten all arrays and stack them into a feature matrix X
    # Shape of X will be (N, num_features)
    flattened_arrays = [arr.flatten() for arr in embedding_arrays]
    X = np.column_stack(flattened_arrays)
    
    # Initialize EM via Gaussian Mixture Model
    gmm = GaussianMixture(n_components=2, covariance_type='full', n_init=10, random_state=42)
    
    # Fit the EM model on the embedding feature space
    gmm.fit(X)
    
    # Predict the posterior probabilities: P(Z=1 | X)
    probs = gmm.predict_proba(X)
    
    # Identify the "Correct" cluster. 
    # We assume the "Correct" cluster has a higher mean on the FIRST feature provided.
    # (Make sure you pass your most reliable metric as the first element in the list!)
    mean_comp_0 = gmm.means_[0, 0]
    mean_comp_1 = gmm.means_[1, 0]
    
    if mean_comp_1 > mean_comp_0:
        fused_flat = probs[:, 1]
    else:
        fused_flat = probs[:, 0]
        
    # Reshape back to (n_examples, n_chains)
    return fused_flat.reshape(original_shape)