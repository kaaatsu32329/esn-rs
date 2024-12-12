use nalgebra as na;

use echo_state_network::RLS;

fn main() {
    let x1 = na::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let d1 = na::DVector::from_vec(vec![5.0, 6.0]);

    let x2 = na::DVector::from_vec(vec![2.0, 4.0, 6.0, 8.0]);
    let d2 = na::DVector::from_vec(vec![10.0, 11.0]);

    let mut rls = RLS::new(4, 2, 1.0, 1.0);

    for _ in 0..100 {
        rls.set_data(&x1, &d1);
    }
    println!("{}", rls);

    for _ in 0..100 {
        rls.set_data(&x2, &d2);
    }
    println!("{}", rls);
}
