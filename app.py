import numpy as np
import math
import random

# ------------------------------------------
# STATIC TOKEN DISTRIBUTION
# ------------------------------------------
# Example words and their "logits"
WORDS = ["cloud", "azure", "data", "security", "storage", "firewall", "gateway"]
LOGITS = np.array([4.6, 3.8, 2.9, 2.4, 1.6, 0.3, -0.8])  
# These values could  be anything — they will be converted into probabilities after softmax.

#These create a very sharp distribution after softmax — the top few values dominate.
#LOGITS = np.array([12.4, 9.7, 8.2, 2.1, 1.9, -5.4, -7.2])

#Softmax → probabilities almost equal → temperature/top-p matter more.
#LOGITS = np.array([0.1, 0.2, 0.15, 0.05, 0.0, 0.01, -0.02])

#Softmax shifts everything so the highest value becomes the most likely, no matter if all logits are negative.
#LOGITS = np.array([-1.8, -2.0, -2.5, -2.7, -3.1, -4.0, -4.2])


# ------------------------------------------
# SOFTMAX FUNCTION
# ------------------------------------------
def softmax(logits):
    exps = np.exp(logits - np.max(logits))  # stability trick
    return exps / np.sum(exps)

# ------------------------------------------
# APPLY TEMPERATURE
# ------------------------------------------
def apply_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")
    return logits / temperature

# ------------------------------------------
# APPLY TOP-K
# ------------------------------------------
def apply_top_k(words, probs, k):
    if k <= 0:
        raise ValueError("top_k must be > 0")
    idx_sorted = np.argsort(probs)[::-1]  # descending
    idx_keep = idx_sorted[:k]
    
    new_probs = np.zeros_like(probs)
    new_probs[idx_keep] = probs[idx_keep]
    new_probs /= new_probs.sum()  # renormalize
    return words, new_probs

# ------------------------------------------
# APPLY TOP-P
# ------------------------------------------
def apply_top_p(words, probs, top_p):
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be between 0 and 1")
        
    idx_sorted = np.argsort(probs)[::-1]
    sorted_probs = probs[idx_sorted]
    cumulative = np.cumsum(sorted_probs)

    # Find cutoff index where cumulative prob exceeds top_p
    cutoff = np.searchsorted(cumulative, top_p) + 1
    
    idx_keep = idx_sorted[:cutoff]

    new_probs = np.zeros_like(probs)
    new_probs[idx_keep] = probs[idx_keep]
    new_probs /= new_probs.sum()  # renormalize
    return words, new_probs

# ------------------------------------------
# FINAL SAMPLING
# ------------------------------------------
def sample_word(words, probs):
    return random.choices(words, weights=probs, k=1)[0]

# ------------------------------------------
# DEMO FUNCTION
# ------------------------------------------
def simulate_sampling(temperature=1.0, top_k=None, top_p=None, verbose=True):
    print("\n==============================")
    print(f"Simulation values:")
    print(f"Temperature = {temperature}")
    print(f"Top-K       = {top_k}")
    print(f"Top-P       = {top_p}")
    print("==============================")

    # 1. Apply temperature
    logits_temp = apply_temperature(LOGITS, temperature)

    # 2. Convert to probabilities
    probs = softmax(logits_temp)

    if verbose:
        print(f"\nInitial probabilities (after applied temperature: {temperature})")
        for t, p in zip(WORDS, probs):
            print(f"{t}: {p:.4f}")

    # 3. Apply top-k
    if top_k is not None:
        words, probs = apply_top_k(WORDS, probs, top_k)
        if verbose:
            print(f"\nAfter Top-K ({top_k}):")
            for t, p in zip(words, probs):
                if p > 0:
                    print(f"{t}: {p:.4f}")

    # 4. Apply top-p
    if top_p is not None:
        words, probs = apply_top_p(WORDS, probs, top_p)
        if verbose:
            print(f"\nAfter Top-P ({top_p}):")
            for t, p in zip(words, probs):
                if p > 0:
                    print(f"{t}: {p:.4f}")

    # 5. Sample final token
    final = sample_word(WORDS, probs)
    print("\n🎯 Final sampled word:", final)
    return final

# ------------------------------------------
# EXAMPLES
# ------------------------------------------

def main():
    # # Deterministic, low randomness
    # simulate_sampling(temperature=0.1, top_k=None, top_p=None)

    # # Moderate randomness
    # simulate_sampling(temperature=0.8, top_k=None, top_p=0.9)

    # # Restrictive top-k
    # simulate_sampling(temperature=1.0, top_k=2, top_p=None)

    # # Combined top-k + top-p
    # simulate_sampling(temperature=0.7, top_k=4, top_p=0.8)
    
    simulate_sampling(temperature=1, top_k=5, top_p=0.8)
    
if __name__ == "__main__":
    main()
# ------------------------------------------


