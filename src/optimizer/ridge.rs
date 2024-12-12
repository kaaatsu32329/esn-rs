use nalgebra as na;
use serde::{Deserialize, Serialize};

/// Ridge regression model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ridge {
    beta: f64,
    x_xt: na::DMatrix<f64>,
    d_xt: na::DMatrix<f64>,
}

impl Ridge {
    /// Create a new Ridge regression model.
    /// 'n_x' is the number of input variables and 'n_y' is the number of output variables.
    /// 'beta' is the regularization parameter.
    pub fn new(n_x: u64, n_y: u64, beta: f64) -> Self {
        let x_xt = na::DMatrix::zeros(n_x as usize, n_x as usize);
        let d_xt = na::DMatrix::zeros(n_y as usize, n_x as usize);

        Ridge { beta, x_xt, d_xt }
    }

    /// Update the internal state of the Ridge regression model.
    /// 'x' is the input vector (explanatory variable) and 'd' is the output vector (response variable).
    pub fn set_data(&mut self, x: &na::DVector<f64>, d: &na::DVector<f64>) {
        self.x_xt = self.x_xt.clone() + x.clone() * x.clone().transpose();
        self.d_xt = self.d_xt.clone() + d.clone() * x.clone().transpose();
    }

    /// Fit the Ridge regression model and return the weight matrix.
    pub fn fit(&self) -> na::DMatrix<f64> {
        let n_x = self.x_xt.ncols();
        let x_xt_inv = (self.x_xt.clone() + self.beta * na::DMatrix::identity(n_x, n_x))
            .try_inverse()
            .unwrap();

        self.d_xt.clone() * x_xt_inv
    }
}

impl std::fmt::Display for Ridge {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut displayed = format!("Beta: {}", self.beta);
        displayed.push_str(&format!("\nx_xt:\n{}", self.x_xt));
        displayed.push_str(&format!("\nd_xt:\n{}", self.d_xt));
        write!(f, "{}", displayed)
    }
}
