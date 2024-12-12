use echo_state_network::*;
use rand::prelude::*;

const TRAIN_STEP: usize = 5000;
const TEST_STEP: usize = 100;
const N_X: u64 = 100;
const BETA: f64 = 0.0;

const RANDOM_SEED: u64 = 41;
const TEST_RANDOM_SEED: u64 = 91;

fn main() {
    let (train_input, train_expected_output) = data_gen(TRAIN_STEP, RANDOM_SEED);
    let (test_input, test_expected_output) = data_gen(TEST_STEP, TEST_RANDOM_SEED);

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
    );

    let mut optimizer = Ridge::new(N_X, n_y, BETA);

    model.train(&train_input, &train_expected_output, &mut optimizer);

    let estimated_output = model.estimate(&test_input);

    let (bits_l2_error, bits_l1_error) =
        get_bits_error_rate(estimated_output.clone(), test_expected_output.clone());
    let (l2_error, l1_error) =
        get_error_rate(estimated_output.clone(), test_expected_output.clone());
    println!("Bits Mean Squared Error: {}", bits_l2_error);
    println!("Bits Mean Absolute Error: {}", bits_l1_error);
    println!("Mean Squared Error: {}", l2_error);
    println!("Mean Absolute Error: {}", l1_error);

    let y_estimated = estimated_output.iter().map(|x| x[0]).collect::<Vec<f64>>();
    let y_expected = test_expected_output
        .clone()
        .into_iter()
        .flatten()
        .collect::<Vec<f64>>();

    plotter::plot(
        "XOR",
        (0..TEST_STEP).map(|v| v as f64).collect::<Vec<f64>>(),
        vec![y_expected, y_estimated],
        vec!["Expected".to_string(), "Output".to_string()],
        Some(&path),
    )
    .unwrap();

    write_as_serde(
        model,
        optimizer,
        &train_input,
        &train_expected_output,
        &test_input,
        &test_expected_output,
        estimated_output,
        None,
    );
}

fn data_gen(step: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let tau = 2;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let train_input_vec = (0..step)
        .map(|_| vec![rng.gen_range(0..2) as f64])
        .collect::<Vec<Vec<f64>>>();

    let mut expected_output_vec = vec![vec![0.0]; step];
    for n in tau..step {
        expected_output_vec[n][0] =
            ((train_input_vec[n - 1][0] as u32) ^ (train_input_vec[n - 2][0] as u32)) as f64;
    }

    (train_input_vec, expected_output_vec)
}

fn get_bits_error_rate(
    estimated_output: Vec<Vec<f64>>,
    expected_output: Vec<Vec<f64>>,
) -> (f64, f64) {
    let mut y_tested_binary = vec![0.0; estimated_output.len()];

    for (n, estimated) in estimated_output.iter().enumerate() {
        if estimated[0] > 0.5 {
            y_tested_binary[n] = 1.0;
        } else {
            y_tested_binary[n] = 0.0;
        }
    }

    let expected_output = expected_output.into_iter().flatten().collect::<Vec<f64>>();

    let mse = mean_squared_error(&expected_output, &y_tested_binary);
    let mae = mean_absolute_error(&expected_output, &y_tested_binary);

    (mse, mae)
}

fn get_error_rate(estimated_output: Vec<Vec<f64>>, expected_output: Vec<Vec<f64>>) -> (f64, f64) {
    let estimated_output = estimated_output.iter().map(|x| x[0]).collect::<Vec<f64>>();
    let expected_output = expected_output.into_iter().flatten().collect::<Vec<f64>>();

    let mse = mean_squared_error(&expected_output, &estimated_output);
    let mae = mean_absolute_error(&expected_output, &estimated_output);

    (mse, mae)
}
