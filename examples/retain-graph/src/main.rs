use burn::{backend::{NdArray, Autodiff}, Tensor, tensor::s};

fn main() {
    type B = Autodiff<NdArray<f64>>;
   
    let device = Default::default();
   
    let a = Tensor::<B, 1>::from_data([3.0, 4.0], &device);
    let b = Tensor::<B, 1,>::from_data([5.0, 6.0], &device);
    
    // Declare that we want autodiff
    let a = a.require_grad();
    let b = b.require_grad();
    
    // Form a * b
    let result = a.clone()
        .reshape([2, 1]).matmul(b.clone().reshape([1, 2]));
    let result1 = result.clone().slice(s![0, 0]);
    let result2 = result.clone().slice(s![1, 0]);
   
    // Compute the gradients
    let all_grads1 = result1.backward();
    let all_grads2 = result2.backward();

    // let all_grads1 = result1.backward_retain();
    // let all_grads2 = result2.backward_retain();
   
    // Evaluate the gradient of result1 with respect to a.
    let grad1 = a.grad(&all_grads1).unwrap();
   
    // Print the resulting tensor
    println!("Gradient: {}", grad1);
   
    // Evaluate the gradient of result2 with respect to a.
    let grad2 = a.grad(&all_grads2).unwrap();
   
    // Print the resulting tensor
    println!("Gradient: {}", grad2);
}