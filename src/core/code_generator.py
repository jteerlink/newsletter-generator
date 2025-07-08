"""
Code Generator for AI/ML Newsletter Content
Generates Python code examples and snippets for open source AI tools
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class CodeType(Enum):
    """Types of code examples to generate"""
    BASIC_EXAMPLE = "basic_example"
    TUTORIAL = "tutorial"
    IMPLEMENTATION = "implementation"
    COMPARISON = "comparison"
    OPTIMIZATION = "optimization"

@dataclass
class CodeExample:
    """Represents a code example with metadata"""
    title: str
    description: str
    code: str
    language: str
    framework: str
    complexity: str  # "beginner", "intermediate", "advanced"
    dependencies: List[str]
    explanation: str
    output_example: Optional[str] = None

class AIMLCodeGenerator:
    """Generate Python code examples for AI/ML tools and frameworks"""
    
    def __init__(self):
        self.supported_frameworks = {
            "pytorch": self._generate_pytorch_example,
            "tensorflow": self._generate_tensorflow_example,
            "huggingface": self._generate_huggingface_example,
            "scikit-learn": self._generate_sklearn_example,
            "pandas": self._generate_pandas_example,
            "numpy": self._generate_numpy_example,
        }
        
    def generate_code_example(self, topic: str, framework: str, 
                            code_type: CodeType = CodeType.BASIC_EXAMPLE,
                            complexity: str = "beginner") -> CodeExample:
        """Generate a code example for a specific topic and framework"""
        
        if framework not in self.supported_frameworks:
            raise ValueError(f"Framework {framework} not supported")
        
        generator_func = self.supported_frameworks[framework]
        return generator_func(topic, code_type, complexity)
    
    def _generate_pytorch_example(self, topic: str, code_type: CodeType, 
                                complexity: str) -> CodeExample:
        """Generate PyTorch code examples"""
        
        if "neural network" in topic.lower():
            return self._pytorch_neural_network_example(complexity)
        elif "transformer" in topic.lower():
            return self._pytorch_transformer_example(complexity)
        elif "training" in topic.lower():
            return self._pytorch_training_example(complexity)
        else:
            return self._pytorch_basic_example(topic, complexity)
    
    def _pytorch_neural_network_example(self, complexity: str) -> CodeExample:
        """Generate PyTorch neural network example"""
        
        if complexity == "beginner":
            code = '''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Create model instance
model = SimpleNN(input_size=784, hidden_size=128, output_size=10)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Model architecture: {model}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")'''
            
            explanation = """This example demonstrates a basic neural network in PyTorch:
- We define a simple feedforward neural network with one hidden layer
- The model inherits from nn.Module, which is the base class for all neural network modules
- The forward method defines how data flows through the network
- We use ReLU activation function and CrossEntropyLoss for classification
- Adam optimizer is used for training with a learning rate of 0.001"""
            
            return CodeExample(
                title="Simple Neural Network in PyTorch",
                description="A basic feedforward neural network for classification tasks",
                code=code,
                language="python",
                framework="pytorch",
                complexity=complexity,
                dependencies=["torch", "torch.nn", "torch.optim"],
                explanation=explanation
            )
        
        # Add more complexity levels as needed
        return self._pytorch_basic_example("neural network", complexity)
    
    def _pytorch_basic_example(self, topic: str, complexity: str) -> CodeExample:
        """Generate basic PyTorch example"""
        
        code = '''import torch
import torch.nn as nn

# Create a simple tensor
x = torch.randn(3, 4)
print(f"Tensor shape: {x.shape}")
print(f"Tensor data: {x}")

# Basic operations
y = torch.matmul(x, x.T)
print(f"Matrix multiplication result: {y}")

# Gradient computation
x.requires_grad_(True)
z = x.sum()
z.backward()
print(f"Gradients: {x.grad}")'''
        
        return CodeExample(
            title=f"Basic PyTorch Example - {topic}",
            description=f"Introduction to PyTorch tensors and operations for {topic}",
            code=code,
            language="python",
            framework="pytorch",
            complexity=complexity,
            dependencies=["torch"],
            explanation="This example shows basic PyTorch tensor operations and gradient computation."
        )
    
    def _generate_tensorflow_example(self, topic: str, code_type: CodeType, 
                                   complexity: str) -> CodeExample:
        """Generate TensorFlow code examples"""
        
        code = '''import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create a simple model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display model architecture
model.summary()

# Generate sample data
x_train = tf.random.normal((1000, 784))
y_train = tf.random.uniform((1000,), maxval=10, dtype=tf.int32)

# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
'''
        
        return CodeExample(
            title=f"TensorFlow Example - {topic}",
            description=f"Building and training a model with TensorFlow for {topic}",
            code=code,
            language="python",
            framework="tensorflow",
            complexity=complexity,
            dependencies=["tensorflow", "tensorflow.keras"],
            explanation="This example demonstrates creating, compiling, and training a neural network with TensorFlow."
        )
    
    def _generate_huggingface_example(self, topic: str, code_type: CodeType, 
                                    complexity: str) -> CodeExample:
        """Generate Hugging Face code examples"""
        
        code = '''from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Tokenize input text
text = "This is an example sentence for BERT processing."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Get model outputs
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

print(f"Input text: {text}")
print(f"Tokenized input shape: {inputs['input_ids'].shape}")
print(f"Output shape: {last_hidden_states.shape}")

# Extract embeddings
sentence_embedding = last_hidden_states.mean(dim=1)
print(f"Sentence embedding shape: {sentence_embedding.shape}")
'''
        
        return CodeExample(
            title=f"Hugging Face Transformers - {topic}",
            description=f"Using pre-trained transformers for {topic}",
            code=code,
            language="python",
            framework="huggingface",
            complexity=complexity,
            dependencies=["transformers", "torch"],
            explanation="This example shows how to use pre-trained BERT model from Hugging Face for text processing."
        )
    
    def _generate_sklearn_example(self, topic: str, code_type: CodeType, 
                                complexity: str) -> CodeExample:
        """Generate scikit-learn code examples"""
        
        code = '''from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = model.feature_importances_
print(f"Top 5 important features: {np.argsort(feature_importance)[-5:]}")
'''
        
        return CodeExample(
            title=f"Scikit-learn Example - {topic}",
            description=f"Machine learning with scikit-learn for {topic}",
            code=code,
            language="python",
            framework="scikit-learn",
            complexity=complexity,
            dependencies=["scikit-learn", "numpy"],
            explanation="This example demonstrates a complete ML pipeline with scikit-learn including data generation, training, and evaluation."
        )
    
    def _generate_pandas_example(self, topic: str, code_type: CodeType, 
                               complexity: str) -> CodeExample:
        """Generate pandas code examples"""
        
        code = '''import pandas as pd
import numpy as np

# Create sample dataset
np.random.seed(42)
data = {
    'feature_1': np.random.normal(0, 1, 1000),
    'feature_2': np.random.normal(2, 1.5, 1000),
    'feature_3': np.random.uniform(0, 10, 1000),
    'target': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
}

df = pd.DataFrame(data)

# Basic data exploration
print("Dataset shape:", df.shape)
print("\\nFirst 5 rows:")
print(df.head())

print("\\nBasic statistics:")
print(df.describe())

# Data preprocessing
df['feature_1_normalized'] = (df['feature_1'] - df['feature_1'].mean()) / df['feature_1'].std()
df['feature_interaction'] = df['feature_1'] * df['feature_2']

# Group analysis
target_analysis = df.groupby('target').agg({
    'feature_1': ['mean', 'std'],
    'feature_2': ['mean', 'std'],
    'feature_3': ['mean', 'std']
}).round(4)

print("\\nTarget group analysis:")
print(target_analysis)
'''
        
        return CodeExample(
            title=f"Pandas Data Analysis - {topic}",
            description=f"Data manipulation and analysis with pandas for {topic}",
            code=code,
            language="python",
            framework="pandas",
            complexity=complexity,
            dependencies=["pandas", "numpy"],
            explanation="This example shows data creation, exploration, preprocessing, and analysis using pandas."
        )
    
    def _generate_numpy_example(self, topic: str, code_type: CodeType, 
                              complexity: str) -> CodeExample:
        """Generate numpy code examples"""
        
        code = '''import numpy as np

# Create sample data
np.random.seed(42)
data = np.random.normal(0, 1, (1000, 10))

print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")

# Basic statistics
print(f"Mean: {np.mean(data, axis=0)}")
print(f"Standard deviation: {np.std(data, axis=0)}")

# Matrix operations
covariance_matrix = np.cov(data, rowvar=False)
print(f"Covariance matrix shape: {covariance_matrix.shape}")

# Eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
print(f"Number of eigenvalues: {len(eigenvalues)}")
print(f"Largest eigenvalue: {np.max(eigenvalues):.4f}")

# Dimensionality reduction (PCA-like)
normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
reduced_data = np.dot(normalized_data, eigenvectors[:, :3])
print(f"Reduced data shape: {reduced_data.shape}")
'''
        
        return CodeExample(
            title=f"NumPy Numerical Computing - {topic}",
            description=f"Numerical computing with NumPy for {topic}",
            code=code,
            language="python",
            framework="numpy",
            complexity=complexity,
            dependencies=["numpy"],
            explanation="This example demonstrates numerical computing with NumPy including statistics, linear algebra, and dimensionality reduction."
        )
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """Validate generated code for syntax and best practices"""
        
        validation_result = {
            "syntax_valid": True,
            "has_imports": False,
            "has_comments": False,
            "follows_pep8": True,
            "issues": [],
            "score": 10.0
        }
        
        # Check for imports
        if "import" in code:
            validation_result["has_imports"] = True
        else:
            validation_result["issues"].append("No import statements found")
            validation_result["score"] -= 1.0
        
        # Check for comments
        if "#" in code or '"""' in code:
            validation_result["has_comments"] = True
        else:
            validation_result["issues"].append("No comments found")
            validation_result["score"] -= 1.0
        
        # Basic syntax validation (simplified)
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            validation_result["syntax_valid"] = False
            validation_result["issues"].append(f"Syntax error: {e}")
            validation_result["score"] -= 3.0
        
        return validation_result
    
    def format_code_for_newsletter(self, code_example: CodeExample) -> str:
        """Format code example for newsletter inclusion"""
        
        formatted = f"""
## {code_example.title}

{code_example.description}

**Framework:** {code_example.framework}  
**Complexity:** {code_example.complexity}  
**Dependencies:** {', '.join(code_example.dependencies)}

```python
{code_example.code}
```

### Explanation

{code_example.explanation}
"""
        
        if code_example.output_example:
            formatted += f"\n### Expected Output\n```\n{code_example.output_example}\n```\n"
        
        return formatted
    
    def suggest_framework(self, topic: str) -> str:
        """Suggest the most appropriate framework for a topic"""
        
        topic_lower = topic.lower()
        
        # Deep learning patterns
        if any(keyword in topic_lower for keyword in [
            "deep learning", "neural network", "cnn", "rnn", "lstm", "transformer"
        ]):
            return "pytorch"
        
        # NLP patterns
        elif any(keyword in topic_lower for keyword in [
            "nlp", "text", "language", "bert", "gpt", "tokenization"
        ]):
            return "huggingface"
        
        # Traditional ML patterns
        elif any(keyword in topic_lower for keyword in [
            "classification", "regression", "clustering", "ensemble"
        ]):
            return "scikit-learn"
        
        # Data analysis patterns
        elif any(keyword in topic_lower for keyword in [
            "data analysis", "preprocessing", "visualization", "pandas"
        ]):
            return "pandas"
        
        # Numerical computing patterns
        elif any(keyword in topic_lower for keyword in [
            "matrix", "linear algebra", "statistics", "numerical"
        ]):
            return "numpy"
        
        # Default to PyTorch for AI/ML topics
        return "pytorch"
    
    def get_supported_frameworks(self) -> List[str]:
        """Get list of supported frameworks"""
        return list(self.supported_frameworks.keys()) 