use echo_state_network::{plotter, EchoStateNetwork, Regularization};
use nalgebra as na;
use rand::prelude::*;

fn main() {
    println!("Start XOR example...");
    let step = 50;
    let input_vec = (0..step)
        .map(|_| thread_rng().gen_range(0..2) as f64)
        .collect::<Vec<f64>>();

    let tau = 2;

    let mut d = vec![0.0; step];
    for n in tau..step {
        d[n] = ((input_vec[n - tau] as u32) ^ (input_vec[n] as u32)) as f64;
    }

    let expected = d.clone();

    let train_input = na::DMatrix::from_column_slice(1, step, &input_vec);

    let n_x = 20;

    let d = na::DMatrix::from_column_slice(1, step, &d.as_slice());

    println!("Set up Echo State Network...");
    let mut model = EchoStateNetwork::new(
        train_input.nrows() as u64,
        d.nrows() as u64,
        n_x,
        0.1,
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

    println!("Train Echo State Network...");
    model.train(
        &train_input,
        &d,
        &mut Regularization::new(n_x, d.nrows() as u64, 0.0),
    );

    let y_trained = model.predict(&train_input);

    let mut y_trained_binary = vec![0.0; step - tau];

    for n in 0..(step - tau) {
        if y_trained[n][0] > 0.5 {
            y_trained_binary[n] = 1.0;
        } else {
            y_trained_binary[n] = 0.0;
        }
    }

    // TODO:
    // let ber = todo!();

    let y_predicted = y_trained.iter().map(|x| x[0]).collect::<Vec<f64>>();

    plotter::plot(
        "XOR",
        (0..step).map(|v| v as f64).collect::<Vec<f64>>(),
        vec![expected, y_predicted, y_trained_binary],
        vec![
            "Expected".to_string(),
            "Output".to_string(),
            "Binary output".to_string(),
        ],
    )
    .unwrap();
}
