use nalgebra as na;

use crate::*;

pub struct EchoStateNetwork {
    input: Input,
    reservoir: Reservoir,
    output: Output,
    previous_y: na::DVector<f64>,
    output_function: fn(&na::DVector<f64>) -> na::DVector<f64>,
    inverse_output_function: fn(&na::DVector<f64>) -> na::DVector<f64>,
    is_classification: bool,
    n_y: u64,
    n_u: u64,
    feedback: Option<Feedback>,
    is_noisy: bool,
    rls: Option<RLS>,
    ridge: Option<Ridge>,
}

impl EchoStateNetwork {
    #[allow(clippy::too_many_arguments)]
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
        ridge_beta: f64,
    ) -> Self {
        EchoStateNetwork {
            input: Input::new(n_u, n_x, input_scale),
            reservoir: Reservoir::new(n_x, density, rho, activation, leaking_rate, None),
            output: Output::new(n_y, n_x),
            previous_y: na::DVector::zeros(n_y as usize),
            output_function,
            inverse_output_function,
            is_classification,
            n_y,
            n_u,
            feedback: feedback_scale.map(|scale| Feedback::new(n_y, n_x, scale)),
            is_noisy: noise_level.is_some(),
            rls: Some(RLS::new(n_x, n_y, 1.0, 1.0)),
            ridge: Some(Ridge::new(n_x, n_y, ridge_beta)),
        }
    }

    pub fn serde_json(&self) -> serde_json::Result<String> {
        let input = serde_json::to_string(&self.input)?;
        let reservoir = serde_json::to_string(&self.reservoir)?;
        let output = serde_json::to_string(&self.output)?;
        let feedback = if let Some(fdb) = self.feedback.clone() {
            serde_json::to_string(&fdb)?
        } else {
            "None".to_string()
        };
        let json = format!(
            r#"{{
            "input": {},
            "reservoir": {},
            "output": {},
            "feedback": {}
            }}"#,
            input, reservoir, output, feedback
        );
        Ok(json)
    }
}

impl ReservoirComputing for EchoStateNetwork {
    fn train(&mut self, teaching_input: &[f64], teaching_output: &[f64]) {
        if self.rls.is_none() {
            panic!("RLS is not initialized");
        }

        let x = na::DVector::from_vec(teaching_input.to_vec());
        let d = na::DVector::from_vec(teaching_output.to_vec());

        self.rls.as_mut().unwrap().set_data(&x, &d);
        let weight = self.rls.as_ref().unwrap().fit();
        self.output.set_weight(weight);
    }

    fn offline_train(&mut self, teaching_input: &[Vec<f64>], teaching_output: &[Vec<f64>]) {
        if self.ridge.is_none() {
            panic!("Ridge is not initialized");
        }

        let train_length = teaching_input.len();
        let input_elements = teaching_input
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<f64>>();
        let teaching_input = na::DMatrix::from_column_slice(
            self.n_u as usize,
            train_length,
            input_elements.as_slice(),
        );
        let output_elements = teaching_output
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<f64>>();
        let teaching_output = na::DMatrix::from_column_slice(
            self.n_y as usize,
            train_length,
            output_elements.as_slice(),
        );

        for n in 0..train_length {
            let mut x_in = self.input.call(&teaching_input.column(n).clone_owned());

            if let Some(fdb) = self.feedback.clone() {
                let x_fdb = fdb.give_feedback(&self.previous_y);
                x_in += x_fdb;
            }

            if self.is_noisy {
                todo!()
            }

            let x_res = self.reservoir.call(x_in);

            if self.is_classification {
                todo!()
            }

            let d = teaching_output.column(n).clone_owned();
            let d = (self.inverse_output_function)(&d);

            self.ridge.as_mut().unwrap().set_data(&x_res, &d);

            self.previous_y = d.clone();
        }

        let output_weight = self.ridge.as_mut().unwrap().fit();
        self.output.set_weight(output_weight);
    }

    fn estimate(&mut self, input: &[f64]) -> Vec<f64> {
        let input = na::DVector::from_column_slice(input);

        let mut x_in = self.input.call(&input);

        if let Some(fdb) = self.feedback.clone() {
            let x_fdb = fdb.give_feedback(&self.previous_y);
            x_in += x_fdb;
        }

        let x_res = self.reservoir.call(x_in);

        if self.is_classification {
            todo!()
        }

        let y_estimated = self.output.call(&x_res);
        let y_estimated = (self.output_function)(&y_estimated);

        self.previous_y = y_estimated.clone();

        y_estimated.as_slice().to_vec()
    }
}
