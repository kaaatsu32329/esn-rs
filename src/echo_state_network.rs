use std::vec;

use nalgebra as na;

use crate::*;

pub struct EchoStateNetwork {
    input: Input,
    reservoir: Reservoir,
    output: Output,

    n_u: u64,
    n_y: u64,
    n_x: u64,
    previous_y: na::DVector<f64>,
    output_function: fn(&na::DVector<f64>) -> na::DVector<f64>,
    inverse_output_function: fn(&na::DVector<f64>) -> na::DVector<f64>,
    is_classification: bool,

    feedback: Option<Feedback>,
    is_noisy: bool,
}

impl EchoStateNetwork {
    pub fn new(
        n_u: u64,
        n_y: u64,
        n_x: u64,
        density: f64,
        input_scale: f64,
        rho: f64,
        activation: fn(f64) -> f64,
        feedback_scale: Option<f64>,
        noise_level: Option<f64>,
        leaking_rate: f64,
        output_function: fn(&na::DVector<f64>) -> na::DVector<f64>,
        inverse_output_function: fn(&na::DVector<f64>) -> na::DVector<f64>,
        is_classification: bool,
    ) -> Self {
        EchoStateNetwork {
            input: Input::new(n_u, n_x, input_scale),
            reservoir: Reservoir::new(n_x, density, rho, activation, leaking_rate, None),
            output: Output::new(n_x, n_y),
            n_u,
            n_y,
            n_x,
            previous_y: na::DVector::zeros(n_y as usize),
            output_function,
            inverse_output_function,
            is_classification,
            feedback: feedback_scale.map(|scale| Feedback::new(n_y, n_x, scale)),
            is_noisy: noise_level.is_some(),
        }
    }

    pub fn train(
        &mut self,
        teaching_input: &na::DMatrix<f64>,
        teaching_output: &na::DMatrix<f64>,
        optimizer: &mut Regularization,
    ) -> Vec<na::DVector<f64>> {
        let train_length = teaching_input.ncols();

        let mut y_log = vec![];

        for n in 0..train_length {
            let mut x_in = self.input.call(&teaching_input.column(n).clone_owned());

            if let Some(fdb) = self.feedback.clone() {
                let x_fdb = fdb.call(&self.previous_y);
                x_in += x_fdb;
            }

            // TODO: Add noise

            let x_res = self.reservoir.call(x_in);

            if self.is_classification {
                todo!()
            }

            let d = teaching_output.column(n).clone_owned();
            let d = (self.inverse_output_function)(&d);

            optimizer.call(&x_res, &d);

            let y = self.output.call(&x_res);
            y_log.push(y.clone());
            self.previous_y = y.clone();
        }

        self.output
            .set_weight(optimizer.get_output_weight_optimized());

        y_log
    }

    pub fn predict(&mut self, input: &na::DMatrix<f64>) -> Vec<na::DVector<f64>> {
        let test_length = input.ncols();

        let mut y_log = vec![];

        for n in 0..test_length {
            let mut x_in = self.input.call(&input.column(n).clone_owned());

            if let Some(fdb) = self.feedback.clone() {
                let x_fdb = fdb.call(&self.previous_y);
                x_in += x_fdb;
            }

            let x_res = self.reservoir.call(x_in);

            if self.is_classification {
                todo!()
            }

            let y_predicted = self.output.call(&x_res);
            y_log.push((self.output_function)(&y_predicted));

            self.previous_y = y_predicted;
        }

        y_log
    }

    pub fn run(&mut self, input: &na::DMatrix<f64>) -> Vec<na::DVector<f64>> {
        let test_length = input.ncols();

        let mut y_log = vec![];

        let mut y = input.column(0).clone_owned();

        for _ in 0..test_length {
            let x_in = self.input.call(&y.clone_owned());

            if let Some(fdb) = self.feedback.clone() {
                let x_fdb = fdb.call(&self.previous_y);
                y += x_fdb;
            }

            let x_res = self.reservoir.call(x_in);

            let y_predicted = self.output.call(&x_res);
            y_log.push((self.output_function)(&y_predicted));
            y = y_predicted;
            self.previous_y = y.clone();
        }

        y_log
    }

    pub fn adapt(
        &mut self,
        input: &na::DMatrix<f64>,
        output: &na::DMatrix<f64>,
        optimizer: &mut Regularization,
    ) -> (Vec<na::DVector<f64>>, Vec<f64>) {
        let data_length = input.ncols();

        let mut y_log = vec![];
        let mut output_weight_abs_mean = vec![];

        for n in 0..data_length {
            let mut x_in = self.input.call(&input.column(n).clone_owned());
            let x_res = self.reservoir.call(x_in);
            let d = output.column(n).clone_owned();
            let d = (self.inverse_output_function)(&d);

            optimizer.call(&x_res, &d);
            let output_weight = optimizer.get_output_weight_optimized();

            let y = output_weight.clone_owned() * x_res;
            y_log.push(y.clone());
            output_weight_abs_mean.push(output_weight.abs().mean());
        }

        (y_log, output_weight_abs_mean)
    }
}
