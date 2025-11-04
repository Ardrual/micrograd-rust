use engine::{MLP, value::Value};

fn compute_loss(mlp: &MLP, xs: &[Vec<Value>], ys: &[Value]) -> f64 {
    let mut total_loss = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let pred = mlp.forward(x);
        let pred_val = &pred[0];

        let diff = pred_val.clone() - y.clone();
        let loss = diff.clone() * diff.clone();
        total_loss += loss.data();
    }
    total_loss / xs.len() as f64
}

fn main() {
    // Create a simple MLP: 2 inputs -> 16 hidden neurons -> 16 hidden neurons -> 1 output
    let mlp = MLP::new(2, &[16, 16, 1]);

    // Data: simple function y = x1 + x2
    let xs = vec![
        vec![Value::new(0.0), Value::new(0.0)],
        vec![Value::new(0.0), Value::new(1.0)],
        vec![Value::new(1.0), Value::new(0.0)],
        vec![Value::new(1.0), Value::new(1.0)],
        vec![Value::new(0.5), Value::new(0.5)],
        vec![Value::new(0.2), Value::new(0.3)],
        vec![Value::new(0.7), Value::new(0.8)],
        vec![Value::new(0.1), Value::new(0.9)],
    ];

    let ys = vec![
        Value::new(0.0),   // 0 + 0 = 0
        Value::new(1.0),   // 0 + 1 = 1
        Value::new(1.0),   // 1 + 0 = 1
        Value::new(2.0),   // 1 + 1 = 2
        Value::new(1.0),   // 0.5 + 0.5 = 1.0
        Value::new(0.5),   // 0.2 + 0.3 = 0.5
        Value::new(1.5),   // 0.7 + 0.8 = 1.5
        Value::new(1.0),   // 0.1 + 0.9 = 1.0
    ];

    // Train/test split: 75% train (6 samples), 25% test (2 samples)
    let train_size = 6;
    let train_xs = xs[0..train_size].to_vec();
    let train_ys = ys[0..train_size].to_vec();
    let test_xs = xs[train_size..].to_vec();
    let test_ys = ys[train_size..].to_vec();

    // Training loop
    let learning_rate = 0.01;
    let epochs = 100;

    println!("Starting training loop...");
    println!("Training a neural network to learn: y = x1 + x2\n");
    println!("Train set size: {} | Test set size: {}\n", train_xs.len(), test_xs.len());

    for epoch in 0..epochs {
        // Forward pass and compute loss on training set
        let mut train_loss = 0.0;

        mlp.zero_grad();

        for (x, y) in train_xs.iter().zip(train_ys.iter()) {
            let pred = mlp.forward(x);
            let pred_val = &pred[0]; // Single output

            // Mean squared error loss
            let diff = pred_val.clone() - y.clone();
            let loss = diff.clone() * diff.clone();

            train_loss += loss.data();

            // Accumulate gradients
            loss.backward();
        }

        train_loss /= train_xs.len() as f64;

        // Update weights before computing test loss
        // Simple SGD update
        let params = mlp.parameters();
        for param in params {
            param.update(learning_rate);
        }

        // Compute loss on test set (no gradients needed)
        mlp.zero_grad();
        let test_loss = compute_loss(&mlp, &test_xs, &test_ys);

        // Print loss every 10 epochs
        if epoch % 10 == 0 {
            println!("Epoch {}: Train Loss = {:.6} | Test Loss = {:.6}", epoch, train_loss, test_loss);
        }

        if epoch == epochs - 1 {
            println!("\nFinal Train Loss: {:.6}", train_loss);
            println!("Final Test Loss: {:.6}\n", test_loss);

            // Show training set predictions
            println!("Training Set Predictions:");
            mlp.zero_grad();
            for (x, expected) in train_xs.iter().zip(train_ys.iter()) {
                let pred = mlp.forward(x);
                let pred_val = &pred[0].data();
                let expected_val = expected.data();
                println!(
                    "Input: [{:.1}, {:.1}] -> Predicted: {:.4}, Expected: {:.1}",
                    x[0].data(), x[1].data(), pred_val, expected_val
                );
            }

            // Show test set predictions
            println!("\nTest Set Predictions:");
            mlp.zero_grad();
            for (x, expected) in test_xs.iter().zip(test_ys.iter()) {
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
