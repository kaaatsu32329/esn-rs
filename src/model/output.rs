use nalgebra as na;
use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Output {
    weight: na::DMatrix<f64>,
}

impl Output {
    pub fn new(n_y: u64, n_x: u64) -> Self {
        let size = n_x * n_y;

        let normal = Normal::new(0.0, 1.0).unwrap();
        let elements = (0..size)
            .map(|_| normal.sample(&mut thread_rng()))
            .collect::<Vec<f64>>();

        let weight = na::DMatrix::from_vec(n_y as usize, n_x as usize, elements);

        Output { weight }
    }

    pub fn call(&self, x: &na::DVector<f64>) -> na::DVector<f64> {
        self.weight.clone() * x
    }

    pub fn output_weight(&self) -> &na::DMatrix<f64> {
        &self.weight
    }

    pub fn set_weight(&mut self, weight: na::DMatrix<f64>) {
        self.weight = weight;
    }
}

impl std::fmt::Display for Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Output weight:\n{:5.2}", self.weight)
    }
}
