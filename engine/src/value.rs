use std::ops::{Add, Mul, Sub};
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt;
use std::collections::HashSet;

enum Op {
    Add,
    Mul,
    Pow(f64),
    Relu,
}

impl Op {
    fn backward(&self, out_grad: f64, inputs: &[Value]) -> Vec<f64> {
        match self {
            Op::Add => vec![out_grad, out_grad],
            Op::Mul => {
                let left = inputs[0].data.borrow().data;
                let right = inputs[1].data.borrow().data;
                vec![out_grad * right, out_grad * left]
            }
            Op::Pow(exponent) => {
                let base = inputs[0].data.borrow().data;
                vec![out_grad * exponent * base.powf(exponent - 1.0)]
            }
            Op::Relu => {
                let input_data = inputs[0].data.borrow().data;
                let grad = if input_data > 0.0 { out_grad } else { 0.0 };
                vec![grad]
            }
        }
    }
}

#[derive(Clone)]
pub struct Value {
    data: Rc<RefCell<ValueData>>,
}

struct ValueData {
    data: f64,
    grad: f64,
    _op: Option<Op>,
    _prev: Option<Vec<Value>>,
}

impl Value {
    pub fn new(data: f64) -> Self {
        Value {
            data: Rc::new(RefCell::new(ValueData {
                data,
                grad: 0.0,
                _op: None,
                _prev: None,
            })),
        }
    }

    pub fn pow(self, exponent: f64) -> Value {
        let out = Value::new(self.data.borrow().data.powf(exponent));
        out.data.borrow_mut()._op = Some(Op::Pow(exponent));
        out.data.borrow_mut()._prev = Some(vec![self.clone()]);
        out
    }

    pub fn relu(self) -> Value {
        let out = Value::new(self.data.borrow().data.max(0.0));
        out.data.borrow_mut()._op = Some(Op::Relu);
        out.data.borrow_mut()._prev = Some(vec![self.clone()]);
        out
    }

    fn build_topo(node: Value, visited: &mut HashSet<*const RefCell<ValueData>>, topo: &mut Vec<Value>) {
        let node_ptr = Rc::as_ptr(&node.data);
        if !visited.contains(&node_ptr) {
            visited.insert(node_ptr);
            if let Some(ref parents) = node.data.borrow()._prev {
                for parent in parents {
                    Value::build_topo(parent.clone(), visited, topo);
                }
            }
            topo.push(node);
        }
    }

    pub fn backward(&self) {
        self.data.borrow_mut().grad = 1.0;
        let mut topo: Vec<Value> = Vec::new();
        let mut visited: HashSet<*const RefCell<ValueData>> = HashSet::new();
        Value::build_topo(self.clone(), &mut visited, &mut topo);
        topo.reverse();

        for node in topo {
            if let Some(ref op) = node.data.borrow()._op {
                if let Some(ref parents) = node.data.borrow()._prev {
                    let out_grad = node.data.borrow().grad;
                    let input_grads = op.backward(out_grad, parents);
                    for (parent, grad) in parents.iter().zip(input_grads.iter()) {
                        parent.data.borrow_mut().grad += *grad;
                    }
                }
            }
        }
    }

    fn print_all_grads(&self) {
        let mut topo: Vec<Value> = Vec::new();
        let mut visited: HashSet<*const RefCell<ValueData>> = HashSet::new();
        Value::build_topo(self.clone(), &mut visited, &mut topo);
        topo.reverse();

        for node in topo {
            println!("Value(data: {}, grad: {})", node.data.borrow().data, node.data.borrow().grad);
        }
    }

    pub fn data(&self) -> f64 {
        self.data.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.data.borrow().grad
    }

    pub fn set_data(&self, val: f64) {
        self.data.borrow_mut().data = val;
    }

    pub fn update(&self, learning_rate: f64) {
        let grad = self.grad();
        let new_val = self.data() - learning_rate * grad;
        self.set_data(new_val);
    }

    pub fn zero_grad(&self) {
        self.data.borrow_mut().grad = 0.0;
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data: {}, grad: {})", self.data.borrow().data, self.data.borrow().grad)
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        let out = Value::new(self.data.borrow().data + other.data.borrow().data);
        out.data.borrow_mut()._op = Some(Op::Add);
        out.data.borrow_mut()._prev = Some(vec![self.clone(), other.clone()]);
        out
    }
}

impl Add<f64> for Value {
    type Output = Value;

    fn add(self, other: f64) -> Value {
        self + Value::new(other)
    }
}

impl Add<Value> for f64 {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        Value::new(self) + other
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        let out = Value::new(self.data.borrow().data * other.data.borrow().data);
        out.data.borrow_mut()._op = Some(Op::Mul);
        out.data.borrow_mut()._prev = Some(vec![self.clone(), other.clone()]);
        out
    }
}

impl Mul<f64> for Value {
    type Output = Value;

    fn mul(self, other: f64) -> Value {
        self * Value::new(other)
    }
}

impl Mul<Value> for f64 {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        Value::new(self) * other
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Value {
        self + (other * -1.0)
    }
}