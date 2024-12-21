use nalgebra as na;

use crate::*;

pub struct PhysicalReservoir {
    output: Output,
    rls: Option<RLS>,
    ridge: Option<Ridge>,
}

impl PhysicalReservoir {
    pub fn new(n_y: u64, n_x: u64) -> Self {
        PhysicalReservoir {
            output: Output::new(n_y, n_x),
            rls: Some(RLS::new(n_x, n_y, 1.0, 1.0)),
            ridge: Some(Ridge::new(n_x, n_y, 0.1)),
        }
    }

    /// Create a new PhysicalReservoir with parameters.
    /// n_y: The number of output nodes.
    /// n_x: The number of input(sensor) nodes.
    /// rls_param: Parameters for RLS. (forgetting_factor, regularization_parameter)
    /// ridge_param: Parameter for Ridge regression. Regularization parameter.
    pub fn new_with_param(n_y: u64, n_x: u64, rls_param: (f64, f64), ridge_param: f64) -> Self {
        PhysicalReservoir {
            output: Output::new(n_y, n_x),
            rls: Some(RLS::new(n_x, n_y, rls_param.0, rls_param.1)),
            ridge: Some(Ridge::new(n_x, n_y, ridge_param)),
        }
    }

    pub fn readout_weight(&self) -> &na::DMatrix<f64> {
        self.output.output_weight()
    }
}

impl ReservoirComputing for PhysicalReservoir {
    /// Online training method.
    /// teaching_input: Input data for training. In this case, it is a sensor data from the physical reservoir.
    fn train(&mut self, teaching_input: &[f64], teaching_output: &[f64]) {
        match &mut self.rls {
            Some(rls) => {
                let x = na::DVector::from_vec(teaching_input.to_vec());
                let d = na::DVector::from_vec(teaching_output.to_vec());
                rls.set_data(&x, &d);
                let weight = rls.fit();
                self.output.set_weight(weight);
            }
            None => panic!("RLS is not initialized"),
        }
    }

    fn offline_train(&mut self, teaching_input: &[Vec<f64>], teaching_output: &[Vec<f64>]) {
        match &mut self.ridge {
            Some(ridge) => {
                for (input, output) in teaching_input.iter().zip(teaching_output.iter()) {
                    let x = na::DVector::from_vec(input.clone());
                    let d = na::DVector::from_vec(output.clone());
                    ridge.set_data(&x, &d);
                }

                let weight = ridge.fit();
                self.output.set_weight(weight);
            }
            None => panic!("Ridge is not initialized"),
        }
    }

    /// Estimate method.
    /// input: Input data for estimating. In this case, it is a sensor data from the physical reservoir.
    fn estimate(&mut self, input: &[f64]) -> Vec<f64> {
        let x = na::DVector::from_vec(input.to_vec());
        let output = self.output.call(&x);
        output.data.as_slice().to_vec()
    }
}
