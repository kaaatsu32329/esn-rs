use crate::*;

/// Serialize the output of the echo state network.
/// To save the log of the echo state network, optimizer, and input/output data as a JSON file.
#[allow(clippy::too_many_arguments)]
pub fn write_as_serde(
    model: EchoStateNetwork,
    train_input: &Vec<Vec<f64>>,
    train_expected_output: &Vec<Vec<f64>>,
    test_input: &Vec<Vec<f64>>,
    test_expected_output: &Vec<Vec<f64>>,
    test_estimated_output: Vec<Vec<f64>>,
    path: Option<&str>,
) {
    let model_json = model.serde_json().unwrap();
    let train_input_log = format!("{:?}", train_input);
    let train_expected_output_log = format!("{:?}", train_expected_output);
    let test_input_log = format!("{:?}", test_input);
    let test_expected_output_log = format!("{:?}", test_expected_output);
    let test_estimated_output_log = format!("{:?}", test_estimated_output);

    let output = format!(
        r#"{{"model":{},"train_input":{},"train_expected_output":{},"test_input":{},"test_expected_output":{},"test_estimated_output":{}}}"#,
        model_json,
        train_input_log,
        train_expected_output_log,
        test_input_log,
        test_expected_output_log,
        test_estimated_output_log
    );

    let date = chrono::Local::now();
    let name = format!("esn_log_{}.json", date.format("%Y-%m-%d-%H-%M-%S"));
    let path = match path {
        Some(p) => format!("{}/{}", p, name),
        None => format!("./log/{}", name),
    };
    let _file = std::fs::File::create(&path).unwrap();
    std::fs::write(&path, output).unwrap();
}
