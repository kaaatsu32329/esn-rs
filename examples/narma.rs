use echo_state_network::*;
use rand::prelude::*;

const TRAIN_STEP: usize = 10000;
const TEST_STEP: usize = 500;
const N_X: u64 = 1000;
const BETA: f64 = 0.1;

const NARMA_ALPHA: f64 = 0.4;
const NARMA_BETA: f64 = 0.1;
const NARMA_GAMMA: f64 = 0.1;
const NARMA_DELTA: f64 = 0.1;

const NARMA_STEP: usize = 10;

const RANDOM_SEED: u64 = 42;
const TEST_RANDOM_SEED: u64 = 92;

fn main() {
    let (train_input, train_expected_output) =
        narma_n_data_gen(TRAIN_STEP, RANDOM_SEED, NARMA_STEP);
    let (test_input, test_expected_output) =
        narma_n_data_gen(TEST_STEP, TEST_RANDOM_SEED, NARMA_STEP);

    let path = format!("{}/examples/graph", env!("CARGO_MANIFEST_DIR"));

    let n_u = train_input.first().unwrap().len() as u64;
    let n_y = train_expected_output.first().unwrap().len() as u64;

    let mut model = EchoStateNetwork::new(
        n_u,
        n_y,
        N_X,
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
        BETA,
    );

    model.offline_train(&train_input, &train_expected_output);

    let mut estimated_output = vec![];
    for input in test_input.iter() {
        estimated_output.push(model.estimate(input));
    }

    let (l2_error, l1_error) = get_error_rate(
        estimated_output.clone(),
        test_expected_output.clone(),
        NARMA_STEP,
    );
    println!("Mean Squared Error: {}", l2_error);
    println!("Mean Absolute Error: {}", l1_error);

    let y_estimated = estimated_output.iter().map(|x| x[0]).collect::<Vec<f64>>();
    let y_expected = test_expected_output
        .clone()
        .into_iter()
        .flatten()
        .collect::<Vec<f64>>();

    plotter::plot(
        "NARMA",
        (0..TEST_STEP).map(|v| v as f64).collect::<Vec<f64>>(),
        vec![y_expected, y_estimated],
        vec!["Expected".to_string(), "Estimated".to_string()],
        Some(&path),
    )
    .unwrap();
}

fn narma_n_data_gen(step: usize, seed: u64, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let input_vec = (0..step)
        .map(|_| vec![rng.gen_range(0.0..1.0)])
        .collect::<Vec<Vec<f64>>>();

    let mut output_vec = vec![vec![0.0]; step];

    for i in n..step {
        let mut state_sum = 0.0;
        for j in 0..n {
            state_sum += input_vec[i - j][0];
        }
        output_vec[i][0] = NARMA_ALPHA * output_vec[i - 1][0]
            + NARMA_BETA * output_vec[i - 1][0] * state_sum
            + NARMA_GAMMA * input_vec[i - n + 1][0] * input_vec[i - 1][0]
            + NARMA_DELTA;
    }

    (input_vec, output_vec)
}

fn get_error_rate(
    estimated_output: Vec<Vec<f64>>,
    expected_output: Vec<Vec<f64>>,
    ignore_bits: usize,
) -> (f64, f64) {
    let estimated_output = estimated_output.iter().map(|x| x[0]).collect::<Vec<f64>>();
    let expected_output = expected_output.into_iter().flatten().collect::<Vec<f64>>();

    let mse = mean_squared_error(
        &expected_output[ignore_bits..],
        &estimated_output[ignore_bits..],
    );
    let mae = mean_absolute_error(
        &expected_output[ignore_bits..],
        &estimated_output[ignore_bits..],
    );

    (mse, mae)
}
