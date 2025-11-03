pub mod value;
use value::Value;


struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    activation: bool, // true for ReLU, false for linear
}

impl Neuron {
    fn new(nin: usize, activation: bool) -> Neuron {
        let mut weights = Vec::with_capacity(nin);
        for _ in 0..nin {
            weights.push(Value::new(rand::random::<f64>() * 2.0 - 1.0));
        }
        let bias = Value::new(0.0);
        Neuron { weights, bias, activation }
    }

    fn forward(&self, x: &[Value]) -> Value {
        let mut act = self.bias.clone();
        for (wi, xi) in self.weights.iter().zip(x.iter()) {
            act = act + wi.clone() * xi.clone();
        }
        if self.activation {
            act.relu()
        } else {
            act
        }
    }

    fn zero_grad(&self) {
        for w in &self.weights {
            w.zero_grad();
        }
        self.bias.zero_grad();
    }

    fn parameters(&self) -> Vec<Value> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(nin: usize, nout: usize, activation: bool) -> Layer {
        let mut neurons = Vec::with_capacity(nout);
        for _ in 0..nout {
            neurons.push(Neuron::new(nin, activation));
        }
        Layer { neurons }
    }

    fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }

    fn zero_grad(&self) {
        for neuron in &self.neurons {
            neuron.zero_grad();
        }
    }

    fn parameters(&self) -> Vec<Value> {
        let mut params = Vec::new();
        for neuron in &self.neurons {
            params.extend(neuron.parameters());
        }
        params
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: &[usize]) -> MLP {
        let mut layers = Vec::with_capacity(nouts.len());
        let mut in_size = nin;
        for (i, &nout) in nouts.iter().enumerate() {
            // ReLU for hidden layers, linear (false) for output layer
            let activation = i < nouts.len() - 1;
            layers.push(Layer::new(in_size, nout, activation));
            in_size = nout;
        }
        MLP { layers }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut out = x.to_vec();
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }

    pub fn zero_grad(&self) {
        for layer in &self.layers {
            layer.zero_grad();
        }
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
}