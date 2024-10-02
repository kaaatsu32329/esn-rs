use nalgebra as na;
use rand::prelude::*;
use rand_distr::Uniform;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Input {
    w_input: na::DMatrix<f64>,
}

impl Input {
    pub fn new(n_u: u64, n_x: u64, input_scale: f64) -> Self {
        let size = n_u * n_x;

        let uniform = Uniform::new(-input_scale, input_scale);

        let elements = (0..size)
            .map(|_| uniform.sample(&mut thread_rng()))
            .collect::<Vec<f64>>();

        let w_input = na::DMatrix::from_vec(n_x as usize, n_u as usize, elements);

        log::debug!("Input weight: {:5.2}", w_input);
        Input { w_input }
    }

    pub fn call(&self, u: &na::DVector<f64>) -> na::DVector<f64> {
        self.w_input.clone() * u
    }

    pub fn debug_print(&self) {
        log::debug!("Input weight: {:5.2}", self.w_input);
    }
}
