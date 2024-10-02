use std::vec;

use echo_state_network::*;
use nalgebra as na;
use rand::prelude::*;

fn main() {
    env_logger::init();

    log::debug!("Start parity example...");
    let step = 100;

    let n_x = 50;

    let (train_input, train_expected_output) = data_gen(step, 1);

    log::debug!("Set up Echo State Network...");
    let mut model = EchoStateNetwork::new(
        train_input.nrows() as u64,
        train_expected_output.nrows() as u64,
        n_x,
        0.05,
        1.0,
        0.9,
        |x| x.tanh(),
        None,
        None,
        1.0,
        |x| x.clone_owned(),
        |x| x.clone_owned(),
        false,
    );

    let mut optimizer = Regularization::new(n_x, train_expected_output.nrows() as u64, 0.0);
    log::debug!("Train Echo State Network...");
    model.train(&train_input, &train_expected_output, &mut optimizer);

    let (test_input, test_expected_output) = data_gen(step, 2);

    let predicted_output = model.predict(&test_input);

    let mut y_tested_binary = vec![0.0; predicted_output.len()];

    for (n, predicted) in predicted_output.iter().enumerate() {
        if predicted[0] > 0.5 {
            y_tested_binary[n] = 1.0;
        } else {
            y_tested_binary[n] = 0.0;
        }
    }

    let bits_errro_rate = bits_error_rate(
        &test_expected_output.iter().copied().collect(),
        &y_tested_binary,
    );

    println!("Bits Error Rate: {}", bits_errro_rate);

    let y_predicted = predicted_output.iter().map(|x| x[0]).collect::<Vec<f64>>();
    let y_expected = test_expected_output.iter().copied().collect::<Vec<f64>>();

    plotter::plot(
        "parity",
        (0..step).map(|v| v as f64).collect::<Vec<f64>>(),
        vec![
            y_expected,
            y_predicted,
            // y_tested_binary
        ],
        vec![
            "Expected".to_string(),
            "Output".to_string(),
            // "Binary output".to_string(),
        ],
    )
    .unwrap();

    model.debug_print();
    optimizer.debug_print();
    log::debug!("End parity example...");
}

fn data_gen(step: usize, seed: u64) -> (na::DMatrix<f64>, na::DMatrix<f64>) {
    let tau = 4;
    let bits = 3;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let input_vec = (0..step)
        .map(|_| rng.gen_range(0..2) as f64)
        .collect::<Vec<f64>>();

    let mut output_vec = vec![0.0; step];
    for n in (tau + bits - 1)..step {
        let bits_sum = input_vec[n - tau] + input_vec[n - tau + 1] + input_vec[n - tau + 2];
        output_vec[n] = bits_sum % 2_f64;
    }

    let train_input_mat = na::DMatrix::from_column_slice(1, step, &input_vec);

    let expected_output_mat = na::DMatrix::from_column_slice(1, step, output_vec.as_slice());

    (train_input_mat, expected_output_mat)
}
