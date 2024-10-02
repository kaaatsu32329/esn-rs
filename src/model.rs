use nalgebra as na;
use petgraph::Graph;
use rand::prelude::*;

#[derive(Debug, Clone)]
pub struct Input {
    w_input: na::DMatrix<f64>,
}

impl Input {
    pub fn new(n_u: u64, n_x: u64, input_scale: f64) -> Self {
        let size = n_u * n_x;
        let elements = (0..size)
            .map(|_| thread_rng().gen_range(-input_scale..input_scale))
            .collect::<Vec<f64>>();

        let w_input = na::DMatrix::from_vec(n_x as usize, n_u as usize, elements);

        Input { w_input }
    }

    /// u: Input vector, n_u x 1
    pub fn call(&self, u: &na::DVector<f64>) -> na::DVector<f64> {
        self.w_input.clone() * u
    }

    pub fn input_weight(&self) -> &na::DMatrix<f64> {
        &self.w_input
    }
}

// TODO: Remove the pub(crate) visibility
#[derive(Debug, Clone)]
pub struct Reservoir {
    pub(crate) seed: Option<u64>,
    pub(crate) weight: na::DMatrix<f64>,
    /// Reservoir vector
    pub(crate) x_vector: na::DVector<f64>,
    pub(crate) activation_function: fn(f64) -> f64,
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

        Reservoir {
            seed,
            weight,
            x_vector: na::DVector::zeros(n_x as usize),
            activation_function: activation,
            alpha: leaking_rate,
        }
    }

    /// TODO: Add description for the arguments
    fn make_connection(n_x: u64, density: f64, rho: f64, seed: Option<u64>) -> na::DMatrix<f64> {
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

        // Element multiplication
        weight.component_mul_assign(&rnd_matrix);

        let eigens = weight.complex_eigenvalues();
        let spectral_radius = eigens
            .iter()
            .map(|x| x.norm())
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();

        weight *= rho / spectral_radius;

        weight
    }

    pub fn call(&mut self, x_in: na::DVector<f64>) -> na::DVector<f64> {
        self.x_vector = (1.0 - self.alpha) * self.x_vector.clone()
            + self.alpha
                * (self.weight.clone() * self.x_vector.clone() + x_in)
                    .map(self.activation_function);
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
}

#[derive(Debug, Clone)]
pub struct Output {
    w_output: na::DMatrix<f64>,
}

impl Output {
    pub fn new(n_x: u64, n_y: u64) -> Self {
        let size = n_x * n_y;
        let elements = (0..size)
            .map(|_| thread_rng().gen_range(-1.0..1.0))
            .collect::<Vec<f64>>();

        let w_output = na::DMatrix::from_vec(n_y as usize, n_x as usize, elements);

        Output { w_output }
    }

    pub fn call(&self, x: &na::DVector<f64>) -> na::DVector<f64> {
        self.w_output.clone() * x
    }

    pub fn output_weight(&self) -> &na::DMatrix<f64> {
        &self.w_output
    }

    pub fn set_weight(&mut self, weight: na::DMatrix<f64>) {
        self.w_output = weight;
    }
}

#[derive(Debug, Clone)]
pub struct Feedback {
    w_feedback: na::DMatrix<f64>,
}

impl Feedback {
    pub fn new(n_y: u64, n_x: u64, feedback_scale: f64) -> Self {
        let size = n_x * n_y;
        let elements = (0..size)
            .map(|_| thread_rng().gen_range(-feedback_scale..feedback_scale))
            .collect::<Vec<f64>>();

        let w_feedback = na::DMatrix::from_vec(n_x as usize, n_y as usize, elements);

        Feedback { w_feedback }
    }

    pub fn call(&self, y: &na::DVector<f64>) -> na::DVector<f64> {
        self.w_feedback.clone() * y
    }

    pub fn feedback_weight(&self) -> &na::DMatrix<f64> {
        &self.w_feedback
    }

    pub fn set_weight(&mut self, weight: na::DMatrix<f64>) {
        self.w_feedback = weight;
    }
}

#[derive(Debug, Clone)]
pub struct Regularization {
    beta: f64,
    x_xt: na::DMatrix<f64>,
    d_xt: na::DMatrix<f64>,
    n_x: u64,
}

impl Regularization {
    pub fn new(n_x: u64, n_y: u64, beta: f64) -> Self {
        let x_xt = na::DMatrix::zeros(n_x as usize, n_x as usize);
        let d_xt = na::DMatrix::zeros(n_y as usize, n_x as usize);

        println!("DEBUG: x_xt: {:?}", x_xt.shape());
        println!("DEBUG: d_xt: {:?}", d_xt.shape());

        Regularization {
            beta,
            x_xt,
            d_xt,
            n_x,
        }
    }

    pub fn call(&mut self, x: &na::DVector<f64>, d: &na::DVector<f64>) {
        self.x_xt = self.x_xt.clone() + x.clone() * x.clone().transpose();
        self.d_xt = self.d_xt.clone() + d.clone() * x.clone().transpose();
    }

    pub fn get_output_weight_optimized(&self) -> na::DMatrix<f64> {
        let x_xt_inv = (self.x_xt.clone()
            + self.beta * na::DMatrix::identity(self.n_x as usize, self.n_x as usize))
        .try_inverse()
        .unwrap();

        self.d_xt.clone() * x_xt_inv
    }
}
