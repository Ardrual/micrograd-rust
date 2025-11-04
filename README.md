# micrograd-rust

A minimal Rust implementation of an automatic differentiation engine with a simple neural network library. This is inspired by [micrograd](https://github.com/karpathy/micrograd) and designed for educational purposes.

## Features

- **Automatic Differentiation**: Build computation graphs and compute gradients via backpropagation
- **Value Type**: Scalar values that track their computation history
- **Neural Network**: Multi-layer perceptron (MLP) with configurable architecture
- **Clean API**: Simple, intuitive interface for defining networks and training

## Architecture

### Value

The core of the autodiff engine. A `Value` represents a scalar in a computation graph and tracks:
- Current data (the scalar value)
- Gradient (computed during backpropagation)
- Operation history (how this value was computed)

**Key methods:**
- `Value::new(data)` - Create a new scalar value
- `value.data()` - Get the current value
- `value.grad()` - Get the computed gradient
- `value.backward()` - Compute gradients for all dependencies
- `value.zero_grad()` - Reset gradients to zero
- `value.update(learning_rate)` - Update value via SGD: `new_value = value - lr * grad`
- `value.pow(exp)` - Power operation
- `value.relu()` - ReLU activation

**Operators:**
- `Value + Value` / `Value + f64` / `f64 + Value` - Addition
- `Value * Value` / `Value * f64` / `f64 * Value` - Multiplication
- `Value - Value` - Subtraction

### MLP (Multi-Layer Perceptron)

A fully-connected neural network with configurable layer sizes.

```rust
// Create a network: 2 inputs -> 16 hidden -> 16 hidden -> 1 output
let mlp = MLP::new(2, &[16, 16, 1]);
```

**Key methods:**
- `mlp.new(nin, nouts)` - Create network with `nin` inputs and layer sizes in `nouts`
- `mlp.forward(x)` - Forward pass, returns output values
- `mlp.parameters()` - Get all weights and biases
- `mlp.zero_grad()` - Reset all gradients

**Architecture notes:**
- Hidden layers use ReLU activation
- Output layer uses linear (identity) activation for regression

## Running Examples

The project includes two example programs that demonstrate how to use the library:

### Basic Training Example
Train a neural network with train/test split:

```bash
cargo run --example basic_training
```

This example trains an MLP to learn the function `y = x1 + x2` with:
- 8 training/test samples (6 train, 2 test)
- 2 inputs → 16 hidden → 16 hidden → 1 output
- MSE loss with SGD optimization
- Demonstrates train/test split and loss tracking

### Computation Graph Example
Explore how the autodiff engine builds and backpropagates through computation graphs:

```bash
cargo run --example computation_graph
```

This example demonstrates:
- Simple computation graphs with multiple operations
- ReLU activation in graphs
- Multiple paths through computation graphs
- Computing gradients for complex expressions

## Usage

### Basic Training Loop

```rust
use engine::{MLP, value::Value};

fn main() {
    // Create network
    let mlp = MLP::new(2, &[16, 16, 1]);

    // Training data
    let xs = vec![
        vec![Value::new(0.0), Value::new(0.0)],
        vec![Value::new(0.0), Value::new(1.0)],
        vec![Value::new(1.0), Value::new(0.0)],
        vec![Value::new(1.0), Value::new(1.0)],
    ];

    let ys = vec![
        Value::new(0.0),
        Value::new(1.0),
        Value::new(1.0),
        Value::new(2.0),
    ];

    let learning_rate = 0.01;
    let epochs = 100;

    for epoch in 0..epochs {
        // Forward pass and loss computation
        let mut total_loss = 0.0;
        mlp.zero_grad();

        for (x, y) in xs.iter().zip(ys.iter()) {
            let pred = mlp.forward(x)[0].clone();

            // MSE loss: (pred - y)^2
            let diff = pred - y.clone();
            let loss = diff.clone() * diff;

            total_loss += loss.data();
            loss.backward();
        }

        total_loss /= xs.len() as f64;

        // Update weights via SGD
        for param in mlp.parameters() {
            param.update(learning_rate);
        }

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, total_loss);
        }
    }
}
```

### Custom Computation Graphs

You can build custom computation graphs beyond the neural network:

```rust
use engine::value::Value;

// Simple computation
let x = Value::new(2.0);
let y = Value::new(3.0);

let z = x.clone() * y.clone() + Value::new(1.0);
z.backward();

println!("z = {}", z.data());        // 7.0
println!("dz/dx = {}", x.grad());    // 3.0
println!("dz/dy = {}", y.grad());    // 2.0
```

## Design Principles

1. **Simplicity**: Minimal code complexity, easy to understand and modify
2. **Encapsulation**: Private implementation details, public API only exposes what users need
3. **Idiomatic Rust**: Uses slices instead of owned vectors, proper access patterns
4. **Gradient Management**: Clean API for accessing and updating gradients

## Implementation Notes

- Values use `Rc<RefCell<>>` for shared mutable state (needed for gradient accumulation)
- Topological sort ensures correct backpropagation order
- ReLU on hidden layers provides non-linearity; linear output for regression tasks
- All operations are scalar-based (no batching)

## Limitations

- Single scalar outputs only (not matrix/tensor operations)
- No built-in optimizers beyond basic SGD
- No GPU support
- Educational implementation, not optimized for performance

## References

- [micrograd](https://github.com/karpathy/micrograd) - Python original
- [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
- [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
