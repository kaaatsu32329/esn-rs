use nalgebra as na;
use petgraph::Graph;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Reservoir {
    /// Reservoir weight
    adjacency_matrix: na::DMatrix<f64>,
    /// Reservoir vector
    x_vector: na::DVector<f64>,
    /// Activation function for the reservoir
    #[serde(skip_serializing, skip_deserializing)]
    activation_function: Option<fn(f64) -> f64>,
    /// Leak rate for the reservoir
    alpha: f64,
}

impl Reservoir {
    pub fn new(
        n_x: u64,
        density: f64,
        rho: f64,
        activation: fn(f64) -> f64,
        leaking_rate: f64,
        seed: Option<u64>,
    ) -> Self {
        let adjacency_matrix = Self::create_adjacency_matrix(n_x, density, rho, seed);

        Reservoir {
            adjacency_matrix,
            x_vector: na::DVector::zeros(n_x as usize),
            activation_function: Some(activation),
            alpha: leaking_rate,
        }
    }

    /// Create an adjacency matrix for the reservoir
    fn create_adjacency_matrix(
        n_x: u64,
        density: f64,
        rho: f64,
        seed: Option<u64>,
    ) -> na::DMatrix<f64> {
        let connected_num = ((n_x * (n_x - 1)) as f64 * density * 0.5) as usize;

        let seed = seed.unwrap_or(0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let graph: Graph<(), ()> =
            petgraph_gen::random_gnm_graph(&mut rng, n_x as usize, connected_num);

        let mut adjacency_matrix = Self::graph_to_adjacency_matrix(&graph);

        let scale = 1.0;
        let rnd_elements = (0..n_x * n_x)
            .map(|_| thread_rng().gen_range(-scale..scale))
            .collect::<Vec<f64>>();
        let rnd_matrix = na::DMatrix::from_vec(n_x as usize, n_x as usize, rnd_elements);

        adjacency_matrix.component_mul_assign(&rnd_matrix);

        let eigens = adjacency_matrix.complex_eigenvalues();
        let eigens_norm = eigens.iter().map(|x| x.norm()).collect::<Vec<f64>>();
        let spectral_radius = eigens_norm
            .iter()
            .fold(0.0, |acc, x| if acc > *x { acc } else { *x });

        adjacency_matrix *= rho / spectral_radius;

        adjacency_matrix
    }

    /// Convert a graph to an adjacency matrix
    fn graph_to_adjacency_matrix(graph: &Graph<(), ()>) -> na::DMatrix<f64> {
        let node_count = graph.node_count();
        let mut adjacency_matrix = na::DMatrix::zeros(node_count, node_count);

        for edge_idx in graph.edge_indices() {
            let (source, target) = graph.edge_endpoints(edge_idx).unwrap();
            adjacency_matrix[(source.index(), target.index())] = 1.0;
            adjacency_matrix[(target.index(), source.index())] = 1.0;
        }

        adjacency_matrix
    }

    pub fn call(&mut self, x_in: na::DVector<f64>) -> na::DVector<f64> {
        self.x_vector = (1.0 - self.alpha) * self.x_vector.clone()
            + self.alpha
                * (self.adjacency_matrix.clone() * self.x_vector.clone() + x_in)
                    .map(self.activation_function.unwrap());
        self.x_vector.clone()
    }
}

impl std::fmt::Display for Reservoir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut displayed = format!("Reservoir adjacency matrix:\n{:5.2}", self.adjacency_matrix);
        displayed.push_str(&format!("\n\nReservoir vector:\n{:5.2}", self.x_vector));
        write!(f, "{}", displayed)
    }
}
