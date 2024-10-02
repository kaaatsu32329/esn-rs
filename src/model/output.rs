use nalgebra as na;
use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Output {
    w_output: na::DMatrix<f64>,
}

impl Output {
    pub fn new(n_y: u64, n_x: u64) -> Self {
        let size = n_x * n_y;

        let normal = Normal::new(0.0, 1.0).unwrap();
        let elements = (0..size)
            .map(|_| normal.sample(&mut thread_rng()))
            .collect::<Vec<f64>>();

        let w_output = na::DMatrix::from_vec(n_y as usize, n_x as usize, elements);

        log::debug!("Output weight: {:5.2}", w_output);
        Output { w_output }
    }

    pub fn call(&self, x: &na::DVector<f64>) -> na::DVector<f64> {
        self.w_output.clone() * x
    }

    pub fn output_weight(&self) -> &na::DMatrix<f64> {
        &self.w_output
    }

    pub fn set_weight(&mut self, weight: na::DMatrix<f64>) {
        self.w_output = weight;
    }

    pub fn debug_print(&self) {
        log::debug!("Output weight: {:5.2}", self.w_output);
    }
}
