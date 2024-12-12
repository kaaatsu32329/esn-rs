use nalgebra as na;

use echo_state_network::Ridge;

fn main() {
    let x = na::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let d = na::DVector::from_vec(vec![5.0, 6.0]);

    let mut ridge = Ridge::new(4, 2, 0.1);
    ridge.set_data(&x, &d);
    let weight = ridge.fit();

    println!("{}", ridge);
    println!("Weight:\n{}", weight);
}
