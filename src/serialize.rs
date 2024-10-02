use crate::*;
use nalgebra as na;

pub fn output_serde(
    model: EchoStateNetwork,
    optimizer: Regularization,
    train_input: &na::DMatrix<f64>,
    train_expected_output: &na::DMatrix<f64>,
    test_input: &na::DMatrix<f64>,
    test_expected_output: &na::DMatrix<f64>,
    test_predicted_output: Vec<na::DVector<f64>>,
) {
    let model_json = model.serde_json().unwrap();
    let optimizer_json = serde_json::to_string(&optimizer).unwrap();
    let train_input_log = format!("{:?}", train_input);
    let train_expected_output_log = format!("{:?}", train_expected_output);
    let test_input_log = format!("{:?}", test_input);
    let test_expected_output_log = format!("{:?}", test_expected_output);
    let test_predicted_output_log = format!("{:?}", test_predicted_output);

    let output = format!(
        r#"{{"model":{},"optimizer":{},"train_input":{},"train_expected_output":{},"test_input":{},"test_expected_output":{},"test_predicted_output":{}}}"#,
        model_json,
        optimizer_json,
        train_input_log,
        train_expected_output_log,
        test_input_log,
        test_expected_output_log,
        test_predicted_output_log
    );

    let date = chrono::Local::now();
    let path = format!(
        "{}{}{}{}",
        "./log/",
        "esn_log_",
        date.format("%Y-%m-%d-%H-%M-%S"),
        ".json"
    );
    let _file = std::fs::File::create(&path).unwrap();
    std::fs::write(&path, output).unwrap();
}
