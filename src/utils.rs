/// Get the mean squared error between two vectors
pub fn mean_squared_error(expected: &[f64], estimated: &[f64]) -> f64 {
    if expected.len() != estimated.len() {
        panic!("The length of the expected and estimated vectors must be the same.");
    }
    let squared_error = expected
        .iter()
        .zip(estimated.iter())
        .fold(0.0, |s, (x, y)| s + (x - y).powi(2));
    squared_error / expected.len() as f64
}

/// Get the mean absolute error between two vectors
pub fn mean_absolute_error(expected: &[f64], estimated: &[f64]) -> f64 {
    if expected.len() != estimated.len() {
        panic!("The length of the expected and estimated vectors must be the same.");
    }
    let absolute_error = expected
        .iter()
        .zip(estimated.iter())
        .fold(0.0, |s, (x, y)| s + (x - y).abs());
    absolute_error / expected.len() as f64
}
