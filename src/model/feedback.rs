use nalgebra as na;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feedback {
    w_feedback: na::DMatrix<f64>,
}

impl Feedback {
    pub fn new(n_y: u64, n_x: u64, feedback_scale: f64) -> Self {
        let size = n_x * n_y;
        let elements = (0..size)
            .map(|_| thread_rng().gen_range(-feedback_scale..feedback_scale))
            .collect::<Vec<f64>>();

        let w_feedback = na::DMatrix::from_vec(n_x as usize, n_y as usize, elements);

        log::debug!("Feedback weight: {:5.2}", w_feedback);
        Feedback { w_feedback }
    }

    pub fn call(&self, y: &na::DVector<f64>) -> na::DVector<f64> {
        self.w_feedback.clone() * y
    }

    pub fn feedback_weight(&self) -> &na::DMatrix<f64> {
        &self.w_feedback
    }

    pub fn set_weight(&mut self, weight: na::DMatrix<f64>) {
        self.w_feedback = weight;
    }

    pub fn debug_print(&self) {
        log::debug!("Feedback weight: {:5.2}", self.w_feedback);
    }
}
