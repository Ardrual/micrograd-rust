
use engine::{MLP, value::Value};

fn main() {
    // Create a simple MLP: 2 inputs -> 16 hidden neurons -> 16 hidden neurons -> 1 output
    let mlp = MLP::new(2, &[16, 16, 1]);

    // Training data: simple function y = x1 + x2
    let xs = vec![
        vec![Value::new(0.0), Value::new(0.0)],
        vec![Value::new(0.0), Value::new(1.0)],
        vec![Value::new(1.0), Value::new(0.0)],
        vec![Value::new(1.0), Value::new(1.0)],
    ];

    let ys = vec![
        Value::new(0.0), // 0 + 0 = 0
        Value::new(1.0), // 0 + 1 = 1
        Value::new(1.0), // 1 + 0 = 1
        Value::new(2.0), // 1 + 1 = 2
    ];

    // Training loop
    let learning_rate = 0.01;
    let epochs = 100;

    println!("Starting training loop...");
    println!("Training a neural network to learn: y = x1 + x2\n");

    for epoch in 0..epochs {
        // Forward pass and compute loss
        let mut total_loss = 0.0;

        mlp.zero_grad();

        for (x, y) in xs.iter().zip(ys.iter()) {
            let pred = mlp.forward(x);
            let pred_val = &pred[0]; // Single output

            // Mean squared error loss
            let diff = pred_val.clone() - y.clone();
            let loss = diff.clone() * diff.clone();

            total_loss += loss.data();

            // Accumulate gradients
            loss.backward();
        }

        total_loss /= xs.len() as f64;

        // Print loss every 10 epochs
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, total_loss);
        }

        // Backward pass already done above, now update weights
        // Simple SGD update
        let params = mlp.parameters();
        for param in params {
            param.update(learning_rate);
        }

        if epoch == epochs - 1 {
            println!("\nFinal Loss: {:.6}\n", total_loss);

            // Test predictions
            println!("Final Predictions:");
            mlp.zero_grad();
            for (x, expected) in xs.iter().zip(ys.iter()) {
                let pred = mlp.forward(x);
                let pred_val = &pred[0].data();
                let expected_val = expected.data();
                println!(
                    "Input: [{:.1}, {:.1}] -> Predicted: {:.4}, Expected: {:.1}",
                    x[0].data(), x[1].data(), pred_val, expected_val
                );
            }
        }
    }
}
