# Temperature, Top-K, and Top-P Sampling Demonstration

This project demonstrates the core sampling techniques used in language models and text generation systems. It shows how **temperature**, **top-k**, and **top-p** parameters affect the randomness and diversity of token selection during text generation.

## 🎯 What This Does

The program simulates the token sampling process that happens inside language models like GPT, Claude, or other transformer-based systems. It takes a set of candidate words with their logits (raw model outputs) and applies different sampling strategies to select the next token.

## 🔧 Sampling Techniques Explained

### Temperature Scaling
- **Low temperature (0.1-0.5)**: More deterministic, focused on high-probability tokens
- **High temperature (1.5-2.0)**: More random, gives lower-probability tokens a better chance
- **Temperature = 1.0**: No modification to the original distribution

### Top-K Sampling
- Keeps only the **K** most probable tokens
- Sets all other probabilities to zero
- Example: `top_k=3` keeps only the 3 most likely words

### Top-P (Nucleus) Sampling
- Keeps the smallest set of tokens whose cumulative probability ≥ **P**
- Dynamic vocabulary size based on probability distribution
- Example: `top_p=0.9` keeps tokens that make up 90% of the probability mass

## 🚀 Getting Started

### Prerequisites
- Python 3.6+
- NumPy

### Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install numpy
   ```

### Running the Demo

```bash
python app.py
```

## 📊 Example Output

The program runs several sampling scenarios:

1. **Low temperature (0.1)** - Deterministic selection
2. **Moderate temperature (0.8) + Top-P (0.9)** - Balanced randomness
3. **Top-K (2)** - Limited to 2 most probable words
4. **Combined Top-K (4) + Top-P (0.8) + Temperature (0.7)** - Multiple constraints
5. **No constraints (temperature=1.0)** - Raw probability distribution

Each scenario shows:
- Initial probabilities after temperature scaling
- Modified probabilities after top-k/top-p filtering
- Final sampled word

## 🎮 Customizing the Demo

### Modify the Word Set
Edit the `WORDS` and `LOGITS` arrays in `app.py`:
```python
WORDS = ["your", "custom", "words", "here"]
LOGITS = np.array([4.0, 2.5, 1.0, -0.5])  # Higher values = more likely
```

### Try Different Parameters
Call `simulate_sampling()` with your own values:
```python
simulate_sampling(temperature=0.5, top_k=3, top_p=0.8)
```

### Experiment with Different Distributions
The code includes commented examples of different logit distributions:
- Sharp distribution (high contrast between probabilities)
- Flat distribution (nearly equal probabilities)
- All-negative logits (demonstrates softmax normalization)

## 🧠 Understanding the Code

### Key Functions
- `softmax()`: Converts logits to probabilities
- `apply_temperature()`: Scales logits by temperature
- `apply_top_k()`: Filters to top K tokens
- `apply_top_p()`: Applies nucleus sampling
- `sample_word()`: Final weighted random selection

### Processing Pipeline
1. **Input**: Raw logits from model
2. **Temperature**: Scale logits by temperature parameter
3. **Softmax**: Convert to probabilities
4. **Top-K**: (Optional) Keep only K most probable tokens
5. **Top-P**: (Optional) Keep tokens up to cumulative probability P
6. **Sample**: Weighted random selection from final distribution

## 🎯 Real-World Applications

These sampling techniques are used in:
- **ChatGPT, Claude, Gemini**: Text generation with controllable creativity
- **Code generation tools**: Balancing correctness vs. diversity
- **Creative writing AI**: Adjusting randomness for different writing styles
- **Translation systems**: Controlling output diversity

## 📚 Learn More

- [Hugging Face Guide to Text Generation](https://huggingface.co/blog/how-to-generate)
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)
- [Neural Text Generation with Sampling Strategies](https://arxiv.org/abs/1904.09751)

## 🤝 Contributing

Feel free to:
- Add new sampling techniques
- Implement visualization of probability distributions
- Add more realistic word/logit examples
- Create interactive parameter adjustment

## 📝 License

This project is open source and available under the MIT License.