use super::NodeId;
use crate::{checkpoint::base::Checkpointer, grads::Gradients, graph::Parent};
use alloc::boxed::Box;

/// Backward step for reverse mode autodiff.
pub trait Step: Send + core::fmt::Debug {
    /// Executes the step and consumes it.
    fn step(self: Box<Self>, grads: &mut Gradients, checkpointer: &mut Checkpointer);
    /// Depth of the operation relative to the first node added to a graph.
    fn depth(&self) -> usize;
    /// The node associated to the step.
    fn node(&self) -> NodeId;
    /// The parents of the node associated to the step.
    fn parents(&self) -> &[Parent];
    /// Clones the step as a boxed trait object
    fn clone_box(&self) -> Box<dyn Step>;
}

impl Clone for Box<dyn Step> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

pub type StepBoxed = Box<dyn Step>;
