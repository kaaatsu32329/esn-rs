
use nalgebra as na;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regularization {
    beta: f64,
    x_xt: na::DMatrix<f64>,
    d_xt: na::DMatrix<f64>,
    n_x: u64,
}

impl Regularization {
    pub fn new(n_x: u64, n_y: u64, beta: f64) -> Self {
        let x_xt = na::DMatrix::zeros(n_x as usize, n_x as usize);
        let d_xt = na::DMatrix::zeros(n_y as usize, n_x as usize);

        Regularization {
            beta,
            x_xt,
            d_xt,
            n_x,
        }
    }

    pub fn call(&mut self, x: &na::DVector<f64>, d: &na::DVector<f64>) {
        self.x_xt = self.x_xt.clone() + x.clone() * x.clone().transpose();
        self.d_xt = self.d_xt.clone() + d.clone() * x.clone().transpose();
    }

    pub fn get_output_weight_optimized(&self) -> na::DMatrix<f64> {
        let x_xt_inv = (self.x_xt.clone()
            + self.beta * na::DMatrix::identity(self.n_x as usize, self.n_x as usize))
        .try_inverse()
        .unwrap();

        self.d_xt.clone() * x_xt_inv
    }

    pub fn debug_print(&self) {
        log::debug!("Regularization beta: {:5.2}", self.beta);
        log::debug!("Regularization x_xt: {:5.2}", self.x_xt);
        log::debug!("Regularization d_xt: {:5.2}", self.d_xt);
    }
}
