pub fn bits_error_rate(expected: &Vec<f64>, predicted: &Vec<f64>) -> f64 {
    let bits_error = expected
        .iter()
        .zip(predicted.iter())
        .fold(0.0, |s, (x, y)| s + (x - y).abs());
    bits_error / expected.len() as f64
}
