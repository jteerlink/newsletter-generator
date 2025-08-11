"""
Code Templates Library for AI/ML Newsletter Generation

This module provides a comprehensive library of code templates for different
AI/ML frameworks, patterns, and use cases to support newsletter content generation.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TemplateCategory(Enum):
    """Categories of code templates"""
    BASIC_EXAMPLE = "basic_example"
    TUTORIAL = "tutorial"
    IMPLEMENTATION = "implementation"
    COMPARISON = "comparison"
    OPTIMIZATION = "optimization"
    PRODUCTION = "production"
    DEBUGGING = "debugging"


class Framework(Enum):
    """Supported AI/ML frameworks"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    HUGGINGFACE = "huggingface"
    SKLEARN = "sklearn"
    PANDAS = "pandas"
    NUMPY = "numpy"
    OPENCV = "opencv"
    KERAS = "keras"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class ComplexityLevel(Enum):
    """Complexity levels for code examples"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class CodeTemplate:
    """Represents a code template with metadata"""
    name: str
    description: str
    framework: Framework
    category: TemplateCategory
    complexity: ComplexityLevel
    code: str
    dependencies: List[str]
    explanation: str
    use_cases: List[str]
    parameters: Dict[str, Any] = None
    expected_output: Optional[str] = None
    notes: List[str] = None


class CodeTemplateLibrary:
    """Comprehensive library of AI/ML code templates"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.template_index = self._build_template_index()
    
    def get_template(self, framework: Framework, category: TemplateCategory, 
                    complexity: ComplexityLevel = ComplexityLevel.BEGINNER) -> Optional[CodeTemplate]:
        """Get a specific template by framework, category, and complexity"""
        key = (framework, category, complexity)
        templates = self.template_index.get(key, [])
        return templates[0] if templates else None
    
    def search_templates(self, query: str, framework: Framework = None, 
                        category: TemplateCategory = None) -> List[CodeTemplate]:
        """Search templates by query string"""
        query_lower = query.lower()
        results = []
        
        for template in self.templates:
            # Filter by framework and category if specified
            if framework and template.framework != framework:
                continue
            if category and template.category != category:
                continue
            
            # Search in name, description, and use cases
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower() or
                any(query_lower in use_case.lower() for use_case in template.use_cases)):
                results.append(template)
        
        return results
    
    def get_templates_by_framework(self, framework: Framework) -> List[CodeTemplate]:
        """Get all templates for a specific framework"""
        return [t for t in self.templates if t.framework == framework]
    
    def get_templates_by_category(self, category: TemplateCategory) -> List[CodeTemplate]:
        """Get all templates for a specific category"""
        return [t for t in self.templates if t.category == category]
    
    def _initialize_templates(self) -> List[CodeTemplate]:
        """Initialize the complete template library"""
        templates = []
        
        # PyTorch templates
        templates.extend(self._create_pytorch_templates())
        
        # TensorFlow templates
        templates.extend(self._create_tensorflow_templates())
        
        # Hugging Face templates
        templates.extend(self._create_huggingface_templates())
        
        # Scikit-learn templates
        templates.extend(self._create_sklearn_templates())
        
        # Pandas templates
        templates.extend(self._create_pandas_templates())
        
        # NumPy templates
        templates.extend(self._create_numpy_templates())
        
        return templates
    
    def _create_pytorch_templates(self) -> List[CodeTemplate]:
        """Create PyTorch code templates"""
        templates = []
        
        # Basic Neural Network
        templates.append(CodeTemplate(
            name="Basic Neural Network",
            description="Simple feedforward neural network with PyTorch",
            framework=Framework.PYTORCH,
            category=TemplateCategory.BASIC_EXAMPLE,
            complexity=ComplexityLevel.BEGINNER,
            code='''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Sample data
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))

# Training loop
model.train()
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/5], Loss: {loss.item():.4f}')

print("Training completed!")''',
            dependencies=["torch", "numpy"],
            explanation="""This template demonstrates a basic neural network in PyTorch:
1. Define a simple feedforward network with one hidden layer
2. Use ReLU activation and dropout for regularization
3. Implement a basic training loop with Adam optimizer
4. Track training progress with loss monitoring""",
            use_cases=["Classification tasks", "Learning PyTorch basics", "Neural network prototyping"],
            expected_output="Training loss decreases over epochs"
        ))
        
        # CNN Template
        templates.append(CodeTemplate(
            name="Convolutional Neural Network",
            description="CNN for image classification with PyTorch",
            framework=Framework.PYTORCH,
            category=TemplateCategory.IMPLEMENTATION,
            complexity=ComplexityLevel.INTERMEDIATE,
            code='''import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model and setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
print(f"Using device: {device}")''',
            dependencies=["torch", "torchvision"],
            explanation="""This CNN template includes:
1. Convolutional layers with ReLU activation
2. Max pooling for dimensionality reduction
3. Fully connected layers for classification
4. Dropout for regularization
5. Device selection (CPU/GPU)
6. Data preprocessing with normalization""",
            use_cases=["Image classification", "Computer vision", "Deep learning tutorials"],
            expected_output="Model architecture summary and device information"
        ))
        
        return templates
    
    def _create_tensorflow_templates(self) -> List[CodeTemplate]:
        """Create TensorFlow code templates"""
        templates = []
        
        # Basic Sequential Model
        templates.append(CodeTemplate(
            name="TensorFlow Sequential Model",
            description="Basic sequential model with TensorFlow/Keras",
            framework=Framework.TENSORFLOW,
            category=TemplateCategory.BASIC_EXAMPLE,
            complexity=ComplexityLevel.BEGINNER,
            code='''import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)

# Create a simple sequential model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# Generate sample data
X_train = np.random.normal(0, 1, (1000, 784))
y_train = np.random.randint(0, 10, 1000)
X_val = np.random.normal(0, 1, (200, 784))
y_val = np.random.randint(0, 10, 200)

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=5,
    validation_data=(X_val, y_val),
    verbose=1
)

print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")''',
            dependencies=["tensorflow", "numpy"],
            explanation="""This TensorFlow template demonstrates:
1. Creating a sequential model with Dense layers
2. Adding dropout for regularization
3. Compiling with optimizer, loss, and metrics
4. Training with validation data
5. Monitoring training progress
6. Extracting final performance metrics""",
            use_cases=["Classification", "TensorFlow basics", "Model prototyping"],
            expected_output="Model summary and training progress with metrics"
        ))
        
        return templates
    
    def _create_huggingface_templates(self) -> List[CodeTemplate]:
        """Create Hugging Face Transformers templates"""
        templates = []
        
        # BERT Text Classification
        templates.append(CodeTemplate(
            name="BERT Text Classification",
            description="Text classification using pre-trained BERT",
            framework=Framework.HUGGINGFACE,
            category=TemplateCategory.IMPLEMENTATION,
            complexity=ComplexityLevel.INTERMEDIATE,
            code='''from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
import numpy as np

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2
)

# Sample data
texts = [
    "I love this product, it's amazing!",
    "This is terrible, worst purchase ever.",
    "Pretty good quality for the price.",
    "Not worth the money, disappointed."
]
labels = [1, 0, 1, 0]  # 1: positive, 0: negative

# Tokenize the data
tokenized_data = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# Make predictions
model.eval()
with torch.no_grad():
    outputs = model(**tokenized_data)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_labels = torch.argmax(predictions, dim=-1)

print("Text Classification Results:")
for i, (text, true_label, pred_label, confidence) in enumerate(
    zip(texts, labels, predicted_labels, predictions)
):
    pred_class = "Positive" if pred_label == 1 else "Negative"
    true_class = "Positive" if true_label == 1 else "Negative"
    conf_score = confidence[pred_label].item()
    
    print(f"\\nExample {i+1}:")
    print(f"Text: '{text}'")
    print(f"True: {true_class}, Predicted: {pred_class}")
    print(f"Confidence: {conf_score:.4f}")''',
            dependencies=["transformers", "torch", "numpy"],
            explanation="""This Hugging Face template shows:
1. Loading pre-trained BERT model and tokenizer
2. Tokenizing text data with proper padding/truncation
3. Making predictions with the model
4. Converting logits to probabilities
5. Extracting predicted classes and confidence scores
6. Comparing predictions with true labels""",
            use_cases=["Sentiment analysis", "Text classification", "NLP applications"],
            expected_output="Classification results with confidence scores"
        ))
        
        return templates
    
    def _create_sklearn_templates(self) -> List[CodeTemplate]:
        """Create scikit-learn templates"""
        templates = []
        
        # ML Pipeline Template
        templates.append(CodeTemplate(
            name="Complete ML Pipeline",
            description="End-to-end machine learning pipeline with scikit-learn",
            framework=Framework.SKLEARN,
            category=TemplateCategory.IMPLEMENTATION,
            complexity=ComplexityLevel.INTERMEDIATE,
            code='''from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# Generate sample dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create ML pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# Train the model
print("Training the model...")
pipeline.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Make predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

# Evaluate the model
print("\\nTest Set Performance:")
print(classification_report(y_test, y_pred))

print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pipeline.named_steps['classifier'].feature_importances_
top_features = np.argsort(feature_importance)[-5:]
print(f"\\nTop 5 important features: {top_features}")
print(f"Their importance scores: {feature_importance[top_features]}")''',
            dependencies=["scikit-learn", "numpy", "pandas"],
            explanation="""This comprehensive ML pipeline includes:
1. Data generation with realistic characteristics
2. Train/test split with stratification
3. Pipeline with preprocessing and model
4. Cross-validation for robust evaluation
5. Performance metrics and confusion matrix
6. Feature importance analysis
7. Probability predictions for uncertainty quantification""",
            use_cases=["Classification problems", "ML workflow", "Model evaluation"],
            expected_output="Cross-validation scores, classification report, and feature importance"
        ))
        
        return templates
    
    def _create_pandas_templates(self) -> List[CodeTemplate]:
        """Create pandas data analysis templates"""
        templates = []
        
        # Data Analysis Template
        templates.append(CodeTemplate(
            name="Comprehensive Data Analysis",
            description="Complete data analysis workflow with pandas",
            framework=Framework.PANDAS,
            category=TemplateCategory.TUTORIAL,
            complexity=ComplexityLevel.INTERMEDIATE,
            code='''import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample dataset
np.random.seed(42)
n_samples = 1000

data = {
    'customer_id': range(1, n_samples + 1),
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.normal(50000, 15000, n_samples),
    'purchase_amount': np.random.exponential(100, n_samples),
    'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_samples),
    'date': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
}

df = pd.DataFrame(data)

# Basic data exploration
print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"\\nData types:\\n{df.dtypes}")
print(f"\\nBasic statistics:\\n{df.describe()}")

# Missing values check
print(f"\\nMissing values:\\n{df.isnull().sum()}")

# Data cleaning and preprocessing
df['income'] = df['income'].clip(lower=0)  # Remove negative incomes
df['month'] = df['date'].dt.month
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 100], 
                        labels=['Young', 'Middle', 'Senior', 'Elder'])

# Exploratory data analysis
print("\\nCategory Distribution:")
print(df['category'].value_counts())

print("\\nAge Group Analysis:")
age_analysis = df.groupby('age_group').agg({
    'income': ['mean', 'median'],
    'purchase_amount': ['mean', 'sum'],
    'customer_id': 'count'
}).round(2)
print(age_analysis)

# Monthly trends
monthly_trends = df.groupby('month').agg({
    'purchase_amount': ['sum', 'mean', 'count']
}).round(2)
print(f"\\nMonthly Purchase Trends:\\n{monthly_trends}")

# Advanced analysis: Customer segmentation
df['purchase_per_income'] = df['purchase_amount'] / df['income']
high_value_customers = df[df['purchase_per_income'] > df['purchase_per_income'].quantile(0.8)]

print(f"\\nHigh-value customers: {len(high_value_customers)} ({len(high_value_customers)/len(df)*100:.1f}%)")
print(f"Average age of high-value customers: {high_value_customers['age'].mean():.1f}")''',
            dependencies=["pandas", "numpy"],
            explanation="""This pandas template demonstrates:
1. Dataset creation with realistic business data
2. Basic data exploration and statistics
3. Data cleaning and preprocessing techniques
4. Feature engineering (age groups, derived metrics)
5. Groupby operations for analysis
6. Time-based analysis with datetime operations
7. Customer segmentation and advanced insights""",
            use_cases=["Data exploration", "Business analytics", "Customer analysis"],
            expected_output="Comprehensive data analysis with statistics and insights"
        ))
        
        return templates
    
    def _create_numpy_templates(self) -> List[CodeTemplate]:
        """Create NumPy numerical computing templates"""
        templates = []
        
        # Linear Algebra Template
        templates.append(CodeTemplate(
            name="Linear Algebra Operations",
            description="Essential linear algebra operations with NumPy",
            framework=Framework.NUMPY,
            category=TemplateCategory.TUTORIAL,
            complexity=ComplexityLevel.INTERMEDIATE,
            code='''import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Create sample matrices
A = np.random.randn(5, 5)
B = np.random.randn(5, 3)
x = np.random.randn(5)

print("Matrix Operations Demo")
print("=" * 30)

# Basic matrix operations
print(f"Matrix A shape: {A.shape}")
print(f"Matrix B shape: {B.shape}")
print(f"Vector x shape: {x.shape}")

# Matrix multiplication
C = np.dot(A, B)
print(f"\\nA @ B result shape: {C.shape}")

# Eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\\nEigenvalues: {eigenvalues[:3]}")  # Show first 3
print(f"Largest eigenvalue: {np.max(eigenvalues.real):.4f}")

# Singular Value Decomposition
U, s, Vt = np.linalg.svd(A)
print(f"\\nSVD - U shape: {U.shape}, s shape: {s.shape}, Vt shape: {Vt.shape}")
print(f"Singular values: {s}")

# Matrix inverse and determinant
try:
    A_inv = np.linalg.inv(A)
    det_A = np.linalg.det(A)
    print(f"\\nDeterminant of A: {det_A:.4f}")
    
    # Verify inverse
    identity_check = np.allclose(np.dot(A, A_inv), np.eye(5))
    print(f"A @ A_inv ≈ I: {identity_check}")
except np.linalg.LinAlgError:
    print("\\nMatrix A is singular (not invertible)")

# Solving linear systems
b = np.random.randn(5)
solution = np.linalg.solve(A, b)
print(f"\\nSolution to Ax = b: {solution}")

# Verify solution
verification = np.allclose(np.dot(A, solution), b)
print(f"Verification (A @ solution ≈ b): {verification}")

# Statistical operations
data_matrix = np.random.normal(0, 1, (1000, 10))
print(f"\\nStatistical Analysis:")
print(f"Data shape: {data_matrix.shape}")
print(f"Mean: {np.mean(data_matrix, axis=0)[:3]}")  # First 3 features
print(f"Std: {np.std(data_matrix, axis=0)[:3]}")

# Correlation matrix
correlation = np.corrcoef(data_matrix, rowvar=False)
print(f"Correlation matrix shape: {correlation.shape}")
print(f"Max correlation (off-diagonal): {np.max(correlation - np.eye(10)):.4f}")''',
            dependencies=["numpy", "matplotlib"],
            explanation="""This NumPy template covers:
1. Matrix creation and basic operations
2. Matrix multiplication and dot products
3. Eigenvalue decomposition for dimensionality analysis
4. Singular Value Decomposition (SVD)
5. Matrix inverse and determinant computation
6. Solving linear equation systems
7. Statistical operations on matrices
8. Correlation analysis""",
            use_cases=["Linear algebra", "Numerical computing", "Data preprocessing"],
            expected_output="Matrix operation results and statistical analysis"
        ))
        
        return templates
    
    def _build_template_index(self) -> Dict[Tuple, List[CodeTemplate]]:
        """Build an index for fast template lookup"""
        index = {}
        
        for template in self.templates:
            key = (template.framework, template.category, template.complexity)
            if key not in index:
                index[key] = []
            index[key].append(template)
        
        return index
    
    def get_template_by_name(self, name: str) -> Optional[CodeTemplate]:
        """Get a template by its exact name"""
        for template in self.templates:
            if template.name == name:
                return template
        return None
    
    def list_available_templates(self) -> Dict[str, List[str]]:
        """List all available templates organized by framework"""
        result = {}
        
        for framework in Framework:
            framework_templates = self.get_templates_by_framework(framework)
            result[framework.value] = [t.name for t in framework_templates]
        
        return result
    
    def generate_template_code(self, template: CodeTemplate, 
                              parameters: Dict[str, Any] = None) -> str:
        """Generate code from template with optional parameter substitution"""
        code = template.code
        
        if parameters and template.parameters:
            # Replace template parameters with actual values
            for param_name, param_value in parameters.items():
                if param_name in template.parameters:
                    placeholder = f"{{{param_name}}}"
                    code = code.replace(placeholder, str(param_value))
        
        return code
    
    def create_custom_template(self, name: str, description: str, framework: Framework,
                              category: TemplateCategory, complexity: ComplexityLevel,
                              code: str, dependencies: List[str], explanation: str,
                              use_cases: List[str]) -> CodeTemplate:
        """Create a custom template and add it to the library"""
        template = CodeTemplate(
            name=name,
            description=description,
            framework=framework,
            category=category,
            complexity=complexity,
            code=code,
            dependencies=dependencies,
            explanation=explanation,
            use_cases=use_cases
        )
        
        self.templates.append(template)
        self.template_index = self._build_template_index()  # Rebuild index
        
        return template


# Global template library instance
template_library = CodeTemplateLibrary()


def get_template(framework: str, category: str, complexity: str = "beginner") -> Optional[CodeTemplate]:
    """Convenience function to get a template"""
    try:
        fw = Framework(framework.lower())
        cat = TemplateCategory(category.lower())
        comp = ComplexityLevel(complexity.lower())
        return template_library.get_template(fw, cat, comp)
    except ValueError:
        return None


def search_templates(query: str, framework: str = None) -> List[CodeTemplate]:
    """Convenience function to search templates"""
    fw = None
    if framework:
        try:
            fw = Framework(framework.lower())
        except ValueError:
            pass
    
    return template_library.search_templates(query, fw)


def list_frameworks() -> List[str]:
    """Get list of supported frameworks"""
    return [fw.value for fw in Framework]


def list_categories() -> List[str]:
    """Get list of template categories"""
    return [cat.value for cat in TemplateCategory]
