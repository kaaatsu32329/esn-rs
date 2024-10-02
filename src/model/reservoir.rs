use nalgebra as na;
use petgraph::Graph;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reservoir {
    pub(crate) seed: Option<u64>,
    pub(crate) weight: na::DMatrix<f64>,
    /// Reservoir vector
    pub(crate) x_vector: na::DVector<f64>,
    #[serde(skip_serializing, skip_deserializing)]
    pub(crate) activation_function: Option<fn(f64) -> f64>,
    /// Leak rate
    pub(crate) alpha: f64,
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
        let weight = Self::make_connection(n_x, density, rho, seed);

        log::debug!("Reservoir weight: {:5.2}", weight);
        Reservoir {
            seed,
            weight,
            x_vector: na::DVector::zeros(n_x as usize),
            activation_function: Some(activation),
            alpha: leaking_rate,
        }
    }

    /// TODO: Add description for the arguments
    fn make_connection(n_x: u64, density: f64, rho: f64, _seed: Option<u64>) -> na::DMatrix<f64> {
        let connected_num = ((n_x * (n_x - 1)) as f64 * density * 0.5) as usize;

        let mut rng = rand::thread_rng();
        let graph: Graph<(), ()> =
            petgraph_gen::random_gnm_graph(&mut rng, n_x as usize, connected_num);

        let mut weight = Self::graph_to_adjacency_matrix(&graph);

        let scale = 1.0;
        let rnd_elements = (0..n_x * n_x)
            .map(|_| thread_rng().gen_range(-scale..scale))
            .collect::<Vec<f64>>();
        let rnd_matrix = na::DMatrix::from_vec(n_x as usize, n_x as usize, rnd_elements);

        weight.component_mul_assign(&rnd_matrix);

        let eigens = weight.complex_eigenvalues();
        let eigens_norm = eigens.iter().map(|x| x.norm()).collect::<Vec<f64>>();
        let spectral_radius = eigens_norm
            .iter()
            .fold(0.0, |acc, x| if acc > *x { acc } else { *x });

        weight *= rho / spectral_radius;

        weight
    }

    pub fn call(&mut self, x_in: na::DVector<f64>) -> na::DVector<f64> {
        self.x_vector = (1.0 - self.alpha) * self.x_vector.clone()
            + self.alpha
                * (self.weight.clone() * self.x_vector.clone() + x_in)
                    .map(self.activation_function.unwrap());
        self.x_vector.clone()
    }

    pub fn reset_state(&mut self) {
        self.x_vector.fill(0.0);
    }

    pub fn weight(&self) -> &na::DMatrix<f64> {
        &self.weight
    }

    fn graph_to_adjacency_matrix(graph: &Graph<(), ()>) -> na::DMatrix<f64> {
        let node_count = graph.node_count();
        let mut adj_matrix = na::DMatrix::zeros(node_count, node_count);

        for edge_idx in graph.edge_indices() {
            let (source, target) = graph.edge_endpoints(edge_idx).unwrap();
            adj_matrix[(source.index(), target.index())] = 1.0;
            adj_matrix[(target.index(), source.index())] = 1.0;
        }

        adj_matrix
    }

    pub fn debug_print(&self) {
        log::debug!("Reservoir weight: {:5.2}", self.weight);
    }
}
