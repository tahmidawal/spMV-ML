# Architecture Diagram: SpMV-CNN2D-SparseMatrices-nonorm

## System Overview
```mermaid
graph TB
    subgraph "Data Generation Pipeline"
        A[FEM Mesh Generation] --> B[Sparse Matrix Assembly]
        B --> C[Sinusoidal Input Generation]
        C --> D[SpMV Ground Truth Computation]
        D --> E[ML Dataset Creation]
    end
    
    subgraph "ML Training Pipeline" 
        E --> F[Dense Matrix Models]
        F --> G[Model Training & Validation]
        G --> H[Performance Evaluation]
        H --> I[Results Visualization]
    end
    
    subgraph "Benchmarking & Analysis"
        J[SpMV Benchmarking] --> K[Performance Comparison]
        K --> L[Timing Analysis]
    end
    
    E --> J
    I --> M[Final Results & Plots]
    L --> M
```

## Detailed Component Architecture

### 1. FEM Data Generation Layer
```mermaid
graph LR
    subgraph "FEM Core (fem.py)"
        A1[FEMPoissonSolver] --> A2[Mesh Generation]
        A2 --> A3[Structured Mesh]
        A2 --> A4[Unstructured Mesh]
        A2 --> A5[Adaptive Mesh]
        A3 --> A6[Matrix Assembly]
        A4 --> A6
        A5 --> A6
        A6 --> A7[Sparse K Matrix]
    end
    
    subgraph "Data Generation (generate_ml_training_data_order1.py)"
        B1[Signal Generation] --> B2[Single Frequency]
        B1 --> B3[Multi Frequency]
        B1 --> B4[Modulated Signals]
        B1 --> B5[Chirp Signals]
        B2 --> B6[SpMV Computation]
        B3 --> B6
        B4 --> B6
        B5 --> B6
        B6 --> B7[Dataset Storage]
    end
    
    A7 --> B6
```

### 2. ML Model Architecture
```mermaid
graph TB
    subgraph "Dense SpMV Models"
        C1[Input Vector] --> C2{Model Type}
        
        C2 --> C3[TwoMatrix Model]
        C2 --> C4[SingleMatrix Model] 
        C2 --> C5[Adaptive Model]
        
        subgraph "TwoMatrix: Y = W1 @ X @ W2^T"
            C3 --> C31[Reshape to Matrix m×n]
            C31 --> C32[W1 @ X]
            C32 --> C33[Result @ W2^T]
            C33 --> C34[Reshape to Vector]
            C34 --> C35[Add Bias]
        end
        
        subgraph "SingleMatrix: Y = W @ X"
            C4 --> C41[Reshape to Matrix]
            C41 --> C42[W @ X]
            C42 --> C43[Reshape + Bias]
        end
        
        subgraph "Adaptive: Learned Permutation"
            C5 --> C51[Soft Permutation]
            C51 --> C52[Matrix Transform]
            C52 --> C53[Output Projection]
        end
        
        C35 --> C6[Output Vector]
        C43 --> C6
        C53 --> C6
    end
```

### 3. Training & Evaluation Pipeline
```mermaid
graph LR
    subgraph "Training Process"
        D1[Load Dataset] --> D2[Model Initialization]
        D2 --> D3[Training Loop]
        D3 --> D4[Forward Pass]
        D4 --> D5[Loss Computation]
        D5 --> D6[Backward Pass]
        D6 --> D7[Optimizer Step]
        D7 --> D8{Validation}
        D8 --> D9[Save Best Model]
        D8 --> D3
    end
    
    subgraph "Evaluation"
        E1[Load Trained Model] --> E2[Test Data]
        E2 --> E3[Prediction]
        E3 --> E4[Error Analysis]
        E4 --> E5[Visualization]
    end
    
    D9 --> E1
```

## File Structure & Responsibilities

### Core Components
- **`fem.py`** (53KB, 1325 lines)
  - FEMPoissonSolver class
  - Mesh generation (structured/unstructured/adaptive)
  - Sparse matrix assembly
  - Boundary condition handling

- **`generate_ml_training_data_order1.py`** (34KB, 861 lines)
  - Sinusoidal signal generation
  - Multiple frequency combinations
  - Dataset creation and storage
  - Visualization utilities

- **`train_dense_spmv.py`** (16KB, 455 lines)
  - Dense matrix model definitions
  - Training pipeline
  - Validation and testing
  - Result saving

### Model Variants
```mermaid
graph TB
    A[Dense SpMV Models] --> B[TwoMatrix Model]
    A --> C[SingleMatrix Model]
    A --> D[Adaptive Model]
    
    B --> B1["Parameters: m² + n² + vector_dim"]
    B --> B2["Operation: W1 @ X @ W2^T"]
    
    C --> C1["Parameters: m² + vector_dim"]
    C --> C2["Operation: W @ X"]
    
    D --> D1["Parameters: vector_dim² + m² + vector_dim²"]
    D --> D2["Operation: Permute → Transform → Project"]
```

### Data Flow
```mermaid
sequenceDiagram
    participant FEM as FEM Solver
    participant Gen as Data Generator
    participant ML as ML Dataset
    participant Model as Dense Models
    participant Eval as Evaluation
    
    FEM->>Gen: Sparse Matrix K
    Gen->>Gen: Generate Sinusoidal Inputs
    Gen->>ML: Compute K @ inputs
    ML->>Model: Training Data (input, output)
    Model->>Model: Learn Dense Approximation
    Model->>Eval: Trained Weights
    Eval->>Eval: Performance Analysis
```

## Key Features

### 1. Matrix Reshaping Strategy
- **Problem**: Sparse matrix-vector multiplication → Dense matrix-matrix multiplication
- **Solution**: Reshape vector to matrix, apply dense operations, reshape back
- **Benefits**: GPU efficiency, parallelizable operations

### 2. Multiple Model Architectures
- **TwoMatrix**: Most expressive, highest parameter count
- **SingleMatrix**: Parameter efficient, good performance
- **Adaptive**: Learned optimal reshaping via soft permutation

### 3. Comprehensive Evaluation
- **In-distribution**: Same frequency ranges as training
- **Out-of-distribution**: Higher frequencies for generalization testing
- **Metrics**: MSE, RMSE, relative error, parameter efficiency

### 4. Visualization Pipeline
- **Training curves**: Loss progression, validation metrics
- **Field comparisons**: Ground truth vs predictions
- **Error analysis**: Spatial error distribution
- **Performance plots**: Speed vs accuracy tradeoffs

## Current State (Sept 13, 2025)
- **Latest Results**: `dense_spmv_results_20250913_135903/`
- **Active Models**: Multiple TwoMatrix configurations (16×89, 8×178, 4×356, 2×712)
- **Dataset**: Order-1 FEM with 1500 mixed sinusoidal samples
- **Evaluation**: Both in-distribution and out-of-distribution testing complete

This architecture represents a novel approach to learning sparse matrix-vector multiplication through dense matrix approximations, leveraging GPU efficiency while maintaining high accuracy for FEM applications.

