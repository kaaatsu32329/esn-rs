use nalgebra as na;

use crate::*;

pub struct PhysicalReservoir {
    input_log: Vec<Vec<f64>>,
    reservoir_log: Vec<Vec<f64>>,
    expected_log: Vec<Vec<f64>>,
    output: Output,
    _n_y: u64,
    _n_x: u64,
}

impl PhysicalReservoir {
    pub fn new(n_y: u64, n_x: u64) -> Self {
        PhysicalReservoir {
            input_log: vec![],
            reservoir_log: vec![],
            expected_log: vec![],
            output: Output::new(n_y, n_x),
            _n_y: n_y,
            _n_x: n_x,
        }
    }

    pub fn update(&mut self, input: &[f64], output: &[f64], expected: &[f64]) {
        self.input_log.push(input.to_vec());
        self.reservoir_log.push(output.to_vec());
        self.expected_log.push(expected.to_vec());
    }

    pub fn offline_train(&mut self, optimizer: &mut Ridge) {
        for (res, expected) in self.reservoir_log.iter().zip(self.expected_log.iter()) {
            let x = na::DVector::from_vec(res.clone());
            let d = na::DVector::from_vec(expected.clone());
            optimizer.set_data(&x, &d);
        }

        let weight = optimizer.fit();
        self.output.set_weight(weight);
    }

    pub fn readout_weight(&self) -> &na::DMatrix<f64> {
        self.output.output_weight()
    }
}
