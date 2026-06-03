use super::*;
use burn_tensor::TensorData;

#[test]
fn should_produce_same_gradients_on_repeated_backward() {
    // Two backward passes on the same graph must yield identical gradients.
    let device = Default::default();

    let x = TestAutodiffTensor::<1>::from_floats([1.0, 2.0, 3.0], &device).require_grad();

    // y = sum(x^2), so dy/dx = 2x = [2, 4, 6]
    let y = x.clone().powf_scalar(2.0).sum();

    let grads1 = y.backward_retain();
    let grads2 = y.backward_retain();

    let grad1 = x.grad(&grads1).unwrap();
    let grad2 = x.grad(&grads2).unwrap();

    // Both backward passes over the same graph must agree
    grad1.to_data().assert_eq(&grad2.to_data(), false);
}

#[test]
fn should_match_standard_backward_gradients() {
    // A retain_graph backward must produce the same gradients as a standard backward.
    let device = Default::default();

    let x_retain =
        TestAutodiffTensor::<1>::from_floats([1.0, 2.0, 3.0], &device).require_grad();
    let x_standard =
        TestAutodiffTensor::<1>::from_floats([1.0, 2.0, 3.0], &device).require_grad();

    // y = sum(x * 3), dy/dx = 3 for all elements
    let y_retain = x_retain.clone().mul_scalar(3.0f32).sum();
    let y_standard = x_standard.clone().mul_scalar(3.0f32).sum();

    let grads_retain = y_retain.backward_retain();
    let grads_standard = y_standard.backward();

    let grad_retain = x_retain.grad(&grads_retain).unwrap();
    let grad_standard = x_standard.grad(&grads_standard).unwrap();

    grad_retain
        .to_data()
        .assert_eq(&TensorData::from([3.0_f32, 3.0, 3.0]), false);
    grad_standard
        .to_data()
        .assert_eq(&TensorData::from([3.0_f32, 3.0, 3.0]), false);
}

#[test]
fn should_allow_multiple_backward_after_retain() {
    // Graph survives retain backwards and can still run a final consuming backward.
    let device = Default::default();

    let x = TestAutodiffTensor::<1>::from_floats([2.0, 3.0], &device).require_grad();
    let y = x.clone().mul_scalar(2.0f32).sum();

    // First pass: retain — dy/dx = 2
    let grads1 = y.backward_retain();
    let grad1 = x.grad(&grads1).unwrap();
    grad1
        .to_data()
        .assert_eq(&TensorData::from([2.0_f32, 2.0]), false);

    // Second pass: retain — same result
    let grads2 = y.backward_retain();
    let grad2 = x.grad(&grads2).unwrap();
    grad2
        .to_data()
        .assert_eq(&TensorData::from([2.0_f32, 2.0]), false);

    // Third pass: standard (consuming) — same result
    let grads3 = y.backward();
    let grad3 = x.grad(&grads3).unwrap();
    grad3
        .to_data()
        .assert_eq(&TensorData::from([2.0_f32, 2.0]), false);
}
