use nalgebra as na;
use serde::{Deserialize, Serialize};

/// Recursive Least Squares (RLS) optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLS {
    /// Auxiliary variable
    p: na::DMatrix<f64>,
    /// Forgetting factor
    lambda: f64,
    /// Weight matrix
    weight: na::DMatrix<f64>,
}

impl RLS {
    pub fn new(n_x: u64, n_y: u64, lambda: f64, alpha: f64) -> Self {
        let mut p = na::DMatrix::identity(n_x as usize, n_x as usize);
        p *= 1.0 / alpha;

        let weight = na::DMatrix::zeros(n_y as usize, n_x as usize);

        RLS { p, lambda, weight }
    }

    pub fn set_data(&mut self, x: &na::DVector<f64>, d: &na::DVector<f64>) {
        let p = self.p.clone();

        let p1 = p.clone() / self.lambda;
        let p2 = p.clone() * x.clone() * x.clone().transpose() * p.clone().transpose();
        let p3 = self.lambda + (x.clone().transpose() * p.clone() * x.clone())[(0, 0)];

        self.p = p1 - p2 / p3;

        let y = self.weight.clone() * x.clone();
        let w1 = self.weight.clone();
        let w2 = (1. / self.lambda) * (d.clone() - y.clone()) * (p * x.clone()).transpose();

        self.weight = w1 + w2;
    }

    pub fn fit(&self) -> na::DMatrix<f64> {
        self.weight.clone()
    }
}

impl std::fmt::Display for RLS {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut displayed = format!("Lambda: {}", self.lambda);
        displayed.push_str(&format!("\nP:\n{}", self.p));
        displayed.push_str(&format!("\nWeight:\n{}", self.weight));
        write!(f, "{}", displayed)
    }
}
