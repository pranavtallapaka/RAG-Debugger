"""
Example datasets for testing RAG Quality Debugger.

This file contains multiple realistic examples with different scenarios:
- Example 1: Mixed quality (some relevant, some redundant, some irrelevant)
- Example 2: High redundancy scenario
- Example 3: Low relevance (retrieval failure scenario)
- Example 4: Good quality retrieval
"""

# ============================================================================
# Example 1: Mixed Quality - Machine Learning Topic
# ============================================================================
# This example has:
# - 2 highly relevant chunks (supervised and unsupervised learning)
# - 2 relevant but redundant chunks (more details on same topics)
# - 1 irrelevant chunk (about Python programming)

EXAMPLE_QUERY_1 = "What is the difference between supervised and unsupervised learning in machine learning?"

EXAMPLE_CHUNKS_1 = [
    # Chunk 1: Highly relevant - supervised learning
    "Supervised learning is a type of machine learning where the algorithm learns from labeled training data. The model is trained on input-output pairs, where the correct output (label) is provided for each input example. Common examples include classification tasks like spam detection and regression tasks like predicting house prices. The key advantage is that the model can learn patterns directly from the labeled examples.",
    
    # Chunk 2: Highly relevant - unsupervised learning
    "Unsupervised learning involves training models on data without labeled examples. The algorithm must find hidden patterns or structures in the data on its own. Common techniques include clustering (grouping similar data points) and dimensionality reduction (reducing the number of features). Unlike supervised learning, there are no correct answers provided during training.",
    
    # Chunk 3: Relevant but redundant with Chunk 1 (more details on supervised learning)
    "Supervised learning algorithms require labeled datasets where each example has a known output. Popular algorithms include linear regression, decision trees, random forests, and neural networks. The training process involves minimizing the error between predicted and actual outputs. This approach works well when you have sufficient labeled data and clear objectives.",
    
    # Chunk 4: Relevant but redundant with Chunk 2 (more details on unsupervised learning)
    "Unsupervised learning is useful when you don't have labeled data or want to discover unknown patterns. K-means clustering groups data into clusters based on similarity, while principal component analysis (PCA) reduces dimensionality. These methods are valuable for exploratory data analysis and can reveal insights that weren't initially obvious.",
    
    # Chunk 5: Irrelevant - about Python, not about ML learning types
    "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming. It has a large standard library and is widely used in web development, data science, and automation."
]

# ============================================================================
# Example 2: High Redundancy Scenario - Climate Change Topic
# ============================================================================
# This example has:
# - Multiple chunks with very similar content (should trigger high redundancy warnings)
# - All chunks are relevant but highly redundant

EXAMPLE_QUERY_2 = "What are the main causes of climate change?"

EXAMPLE_CHUNKS_2 = [
    # Chunk 1: Main causes
    "Climate change is primarily caused by greenhouse gas emissions from human activities. The main sources include burning fossil fuels for energy, deforestation, industrial processes, and agricultural practices. Carbon dioxide is the most significant greenhouse gas, accounting for about 76% of total emissions.",
    
    # Chunk 2: Very similar to Chunk 1 (high redundancy)
    "The primary drivers of climate change are greenhouse gases released through human activities. Key contributors are fossil fuel combustion for power generation, forest clearing, manufacturing operations, and farming methods. CO2 represents the largest portion of greenhouse gas emissions, making up approximately three-quarters of the total.",
    
    # Chunk 3: Similar content, slightly different wording
    "Human activities that release greenhouse gases are the main causes of climate change. These activities include using coal, oil, and gas for electricity and transportation, cutting down forests, and various industrial and agricultural processes. Carbon dioxide emissions are the dominant factor, representing around 76% of all greenhouse gas emissions.",
    
    # Chunk 4: More specific but still overlapping
    "Fossil fuel burning is the largest source of greenhouse gas emissions causing climate change. When we burn coal, oil, and natural gas for energy, we release carbon dioxide into the atmosphere. Deforestation also contributes significantly by reducing the number of trees that can absorb CO2.",
    
    # Chunk 5: Different angle but still relevant
    "The greenhouse effect is enhanced by human activities, leading to global warming. Industrial processes, transportation, and energy production from fossil fuels are major contributors. Methane from agriculture and landfills also plays a significant role, though CO2 remains the primary greenhouse gas."
]

# ============================================================================
# Example 3: Low Relevance (Retrieval Failure) - Quantum Computing Topic
# ============================================================================
# This example has:
# - Query about quantum computing
# - Chunks that are mostly irrelevant (about classical computing, general physics)
# - Should trigger missing context warnings

EXAMPLE_QUERY_3 = "How do quantum computers use qubits to perform calculations?"

EXAMPLE_CHUNKS_3 = [
    # Chunk 1: About classical computers (irrelevant)
    "Classical computers use bits that can be either 0 or 1. These bits are processed by transistors in integrated circuits. The CPU executes instructions sequentially, performing calculations using binary arithmetic. Modern processors can execute billions of operations per second.",
    
    # Chunk 2: About general physics (somewhat related but not directly relevant)
    "Quantum mechanics describes the behavior of particles at the atomic and subatomic level. Particles can exist in superposition states, meaning they can be in multiple states simultaneously until measured. This probabilistic nature is fundamental to quantum physics.",
    
    # Chunk 3: About computer history (irrelevant)
    "The first electronic computers were developed in the 1940s. ENIAC, completed in 1945, was one of the earliest general-purpose computers. It used vacuum tubes and could perform calculations much faster than mechanical calculators. Modern computers have evolved significantly from these early machines.",
    
    # Chunk 4: About binary systems (tangentially related)
    "Binary number systems use only two digits, 0 and 1. This base-2 system is fundamental to digital computing. All data in computers is ultimately represented as binary digits. Boolean algebra provides the mathematical foundation for binary operations.",
    
    # Chunk 5: About general computing concepts (irrelevant)
    "Computer algorithms are step-by-step procedures for solving problems. They can be implemented in various programming languages. Algorithm efficiency is measured by time and space complexity. Common algorithms include sorting, searching, and graph traversal techniques."
]

# ============================================================================
# Example 4: Good Quality Retrieval - Neural Networks Topic
# ============================================================================
# This example has:
# - All chunks are highly relevant
# - Good coverage of different aspects
# - Low redundancy (complementary information)
# - Should show good scores across all metrics

EXAMPLE_QUERY_4 = "What are the key components of a neural network and how do they work together?"

EXAMPLE_CHUNKS_4 = [
    # Chunk 1: Neurons and layers
    "Neural networks consist of interconnected nodes called neurons, organized into layers. The input layer receives data, hidden layers process information, and the output layer produces results. Each neuron receives inputs, applies weights, and passes the result through an activation function.",
    
    # Chunk 2: Weights and connections
    "Connections between neurons have associated weights that determine signal strength. During training, these weights are adjusted to minimize prediction errors. The learning process uses algorithms like backpropagation to update weights based on the difference between predicted and actual outputs.",
    
    # Chunk 3: Activation functions
    "Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Common activation functions include ReLU (Rectified Linear Unit), sigmoid, and tanh. These functions determine whether a neuron should be activated based on its input.",
    
    # Chunk 4: Forward and backward propagation
    "Forward propagation passes input data through the network layer by layer to generate predictions. Backward propagation then calculates gradients and updates weights to improve accuracy. This iterative process continues until the model converges to an optimal solution.",
    
    # Chunk 5: Loss functions and optimization
    "Loss functions measure how far predictions are from actual values. Common loss functions include mean squared error for regression and cross-entropy for classification. Optimization algorithms like gradient descent adjust weights to minimize the loss function."
]

# ============================================================================
# Default exports (for backward compatibility)
# ============================================================================
EXAMPLE_QUERY = EXAMPLE_QUERY_1
EXAMPLE_CHUNKS = EXAMPLE_CHUNKS_1

# All examples in a dictionary for easy access
ALL_EXAMPLES = {
    "Example 1: Mixed Quality (ML)": {
        "query": EXAMPLE_QUERY_1,
        "chunks": EXAMPLE_CHUNKS_1,
        "description": "Mixed quality: some relevant, some redundant, some irrelevant"
    },
    "Example 2: High Redundancy (Climate)": {
        "query": EXAMPLE_QUERY_2,
        "chunks": EXAMPLE_CHUNKS_2,
        "description": "High redundancy: multiple similar chunks about climate change"
    },
    "Example 3: Low Relevance (Quantum)": {
        "query": EXAMPLE_QUERY_3,
        "chunks": EXAMPLE_CHUNKS_3,
        "description": "Low relevance: query about quantum computing, chunks about classical computing"
    },
    "Example 4: Good Quality (Neural Networks)": {
        "query": EXAMPLE_QUERY_4,
        "chunks": EXAMPLE_CHUNKS_4,
        "description": "Good quality: all relevant, good coverage, low redundancy"
    }
}
