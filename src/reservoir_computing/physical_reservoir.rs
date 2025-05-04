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
        if self.rls.is_none() {
            panic!("RLS is not initialized");
        }
        let x = na::DVector::from_vec(teaching_input.to_vec());
        let d = na::DVector::from_vec(teaching_output.to_vec());
        self.rls.as_mut().unwrap().set_data(&x, &d);
        let weight = self.rls.as_mut().unwrap().fit();
        self.output.set_weight(weight);
    }

    fn offline_train(&mut self, teaching_input: &[Vec<f64>], teaching_output: &[Vec<f64>]) {
        if self.ridge.is_none() {
            panic!("Ridge is not initialized");
        }

        for (input, output) in teaching_input.iter().zip(teaching_output.iter()) {
            let x = na::DVector::from_vec(input.clone());
            let d = na::DVector::from_vec(output.clone());
            self.ridge.as_mut().unwrap().set_data(&x, &d);
        }

        let weight = self.ridge.as_mut().unwrap().fit();
        self.output.set_weight(weight);
    }

    /// Estimate method.
    /// input: Input data for estimating. In this case, it is a sensor data from the physical reservoir.
    fn estimate(&mut self, input: &[f64]) -> Vec<f64> {
        let x = na::DVector::from_vec(input.to_vec());
        let output = self.output.call(&x);
        output.data.as_slice().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_physical_reservoir() {
        let mut reservoir = PhysicalReservoir::new_with_param(1, 2, (0.0, 0.0), 0.0);
        let teaching_input = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let teaching_output = vec![vec![1.5], vec![3.5]];

        reservoir.offline_train(&teaching_input, &teaching_output);

        let input = vec![5.0, 6.0];
        let output = reservoir.estimate(&input);
        let expected_output = vec![5.5];
        assert_approx_eq!(output[0], expected_output[0]);
    }
}
