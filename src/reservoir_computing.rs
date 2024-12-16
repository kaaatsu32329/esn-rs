mod echo_state_network;
mod physical_reservoir;

pub use echo_state_network::*;
pub use physical_reservoir::*;

pub trait ReservoirComputing {
    /// Online training method.
    fn train(&mut self, teaching_input: &[f64], teaching_output: &[f64]);
    /// Offline training method.
    fn offline_train(&mut self, teaching_input: &[Vec<f64>], teaching_output: &[Vec<f64>]);
    /// Estimate method.
    fn estimate(&mut self, input: &[f64]) -> Vec<f64>;
}
