use engine::value::Value;

fn main() {
    println!("=== Computation Graph and Backpropagation Example ===\n");

    // Example 1: Simple computation graph
    println!("Example 1: Simple computation graph");
    println!("Computing: f(a, b) = a * b + (a ^ 3)\n");

    let a = Value::new(2.0);
    let b = Value::new(3.0);

    // Build computation graph: f = a*b + a^3
    let mul = a.clone() * b.clone();
    let pow = a.clone().pow(3.0);
    let f = mul + pow;

    println!("a = {}", a.data());
    println!("b = {}", b.data());
    println!("f = a*b + a^3 = {} + {} = {}\n", a.data() * b.data(), a.data().powf(3.0), f.data());

    // Backpropagation
    f.backward();

    println!("Gradients after backpropagation:");
    println!("df/da = {} (expected: b + 3*a^2 = 3 + 12 = 15.0)", a.grad());
    println!("df/db = {} (expected: a = 2.0)\n", b.grad());

    // Example 2: More complex graph with ReLU
    println!("Example 2: Computation graph with ReLU activation");
    println!("Computing: f(x) = ReLU(2*x - 1)^2\n");

    let x = Value::new(1.5);

    // Build computation graph: f = ReLU(2*x - 1)^2
    let two_x = x.clone() * Value::new(2.0);
    let shifted = two_x - Value::new(1.0);
    let activated = shifted.relu();
    let f2 = activated.clone().pow(2.0);

    println!("x = {}", x.data());
    println!("2*x - 1 = {}", (2.0 * x.data() - 1.0));
    println!("ReLU(2*x - 1) = {}", activated.data());
    println!("f = ReLU(2*x - 1)^2 = {}\n", f2.data());

    // Backpropagation
    f2.backward();

    println!("Gradient after backpropagation:");
    println!("df/dx = {}\n", x.grad());

    // Example 3: Multiple paths through computation graph
    println!("Example 3: Multiple paths in computation graph");
    println!("Computing: f(x) = x^2 + x + 1\n");

    let x = Value::new(3.0);

    // Build computation graph with multiple uses of x
    let x_squared = x.clone() * x.clone();
    let x_squared_plus_x = x_squared + x.clone();
    let f3 = x_squared_plus_x + Value::new(1.0);

    println!("x = {}", x.data());
    println!("f = x^2 + x + 1 = {} + {} + 1 = {}\n", x.data() * x.data(), x.data(), f3.data());

    // Backpropagation
    f3.backward();

    println!("Gradient after backpropagation:");
    println!("df/dx = {} (expected: 2*x + 1 = 7.0)", x.grad());
}
