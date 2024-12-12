use nalgebra as na;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Feedback {
    weight: na::DMatrix<f64>,
}

impl Feedback {
    pub fn new(n_y: u64, n_x: u64, feedback_scale: f64) -> Self {
        let size = n_x * n_y;
        let elements = (0..size)
            .map(|_| thread_rng().gen_range(-feedback_scale..feedback_scale))
            .collect::<Vec<f64>>();

        let weight = na::DMatrix::from_vec(n_x as usize, n_y as usize, elements);

        Feedback { weight }
    }

    pub fn give_feedback(&self, y: &na::DVector<f64>) -> na::DVector<f64> {
        self.weight.clone() * y
    }
}

impl std::fmt::Display for Feedback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Feedback: {:?}", self.weight)
    }
}
