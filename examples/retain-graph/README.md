# retain-graph

This example demonstrates `retain_graph` support in Burn — the ability to call `backward()` multiple times on the same computational graph without destroying it, equivalent to PyTorch's `loss.backward(retain_graph=True)`.

## Running the Example

```bash
# From the workspace root
cargo run -p retain-graph

# Or from the example directory
cd examples/retain-graph
cargo run
```

## What the Example Shows

The example constructs a graph for `result = a ⊗ b` (outer product), then computes gradients of two different slices of that result:

- `result1 = result[0, 0]` → `a[0] * b`
- `result2 = result[1, 0]` → `a[1] * b`

With standard `backward()`, the first call destroys the graph — the second call panics or produces wrong results. With `backward_retain()`, the graph survives both passes.

Switch between the two modes by toggling the comments in `src/main.rs`.

---

## Implementation: Adding `retain_graph` to Burn

### The Problem

Before this change, calling `backward()` on a tensor permanently destroyed the computational graph. The root cause was ownership transfer throughout the backward execution pipeline:

```
backward()
  steps.remove(&node_id)            // root step consumed from HashMap
  actions_builder.remove(&node_id) // root builder consumed

BreadthFirstSearch::traverse()
  steps.remove(&id)                 // every visited step removed and owned
  callback(step)                    // step moved into tape

execute_steps()
  tape.into_iter()                  // tape consumed
  step.step(grads, checkpointer)    // Step::step(self: Box<Self>) destroys step

cleanup
  free_unavailable_nodes()          // removes remaining steps and builders
```

Every node was taken out of its `HashMap` by value and consumed. After one backward pass, nothing remained.

### Ownership Bottlenecks

Five interconnected ownership bottlenecks made the graph non-reusable:

1. **`Step::step(self: Box<Self>)`** — the trait method consumed the boxed step on execution
2. **`Backward::backward(self, ops)`** — the backward handler and its ops were consumed
3. **`HashMap::remove()` in traversal** — each step was extracted and destroyed
4. **`actions_builder.remove()` in `build_tape`** — checkpoint builders removed per node
5. **`Box<dyn Any + Send>` for checkpoint state** — not cloneable, so builders couldn't be reused

### What Was Changed

#### 1. `Step` trait — `graph/base.rs`

```rust
// Before: step consumes itself
fn step(self: Box<Self>, grads: &mut Gradients, checkpointer: &mut Checkpointer);

// After: step borrows itself
fn step(&self, grads: &mut Gradients, checkpointer: &mut Checkpointer);
```

This is the foundational change. Once `step` borrows instead of consuming, borrowed steps can execute without being destroyed.

#### 2. `OpsStep::step` — `ops/base.rs`

```rust
fn step(&self, grads: &mut Gradients, checkpointer: &mut Checkpointer) {
    let ops = Ops::new(
        self.ops.parents.clone(),
        self.ops.node.clone(),
        self.ops.state.clone(), // S: Clone was already required
    );
    self.backward.backward(ops, grads, checkpointer);
}
```

The state is cloned before forwarding to the backward function. Since `State: Clone` was already a trait bound on `Backward`, this is always valid. For most operations the state contains tensors backed by `Arc`, so the clone is a cheap reference-count increment.

#### 3. `Backward` trait — `ops/backward.rs`

```rust
// Before
fn backward(self, ops: Ops<Self::State, N>, grads: &mut Gradients, checkpointer: &mut Checkpointer);

// After
fn backward(&self, ops: Ops<Self::State, N>, grads: &mut Gradients, checkpointer: &mut Checkpointer);
```

All `Backward` implementations (`~91` across `tensor.rs`, `module.rs`, `activation.rs`, etc.) are **zero-sized types** by design — the Burn documentation explicitly states "Concrete types implementing this trait should not have any state." None of them use `self` in their bodies. Changing `self` to `&self` is therefore purely mechanical and semantically identical.

`CatStep` was a special case: it directly implements `Step` (not `Backward`) and consumed `self.nodes.into_iter()`. Fixed by switching to `self.nodes.iter().cloned()`.

#### 4. Checkpoint state — `checkpoint/state.rs` and `checkpoint/builder.rs`

```rust
// Before
pub(crate) type StateContent = Box<dyn Any + Send>;

// After
pub(crate) type StateContent = Arc<dyn Any + Send>;
```

`CheckpointingAction::Computed` stores a snapshot of a tensor primitive. With `Box`, the snapshot was one-time use. With `Arc`, it can be cheaply shared across multiple backward passes without copying the underlying data.

This also enables `CheckpointingAction` and `CheckpointerBuilder` to implement `Clone`, which is required for the retain path.

A new `extend_ref` method was added to `CheckpointerBuilder`:
```rust
pub(crate) fn extend_ref(&mut self, other: &CheckpointerBuilder) {
    // Clones actions (Arc clone for Computed, Arc clone for Recompute)
}
```

#### 5. Non-destructive traversal — `graph/traversal.rs`

A new `traverse_retaining` method was added alongside the existing `traverse`:

```rust
pub fn traverse_retaining<F, I>(
    &self,
    root_id: NodeId,
    steps: &HashMap<NodeId, I>, // immutable borrow — no removal
    mut callback: F,
) where
    F: FnMut(NodeId, &I), // callback receives reference, not owned value
    I: TraversalItem,
{
    // BFS over steps using get() instead of remove()
}
```

The existing `traverse` (which uses `remove()`) is unchanged for the normal path.

#### 6. Retain path in the server — `runtime/server.rs`

Three new methods were added to `AutodiffServer`:

- **`build_tape_retaining`** — builds `Vec<Vec<NodeId>>` (IDs only) by borrowing steps and checkpoint builders via `extend_ref`. No nodes are consumed.
- **`execute_steps_retaining`** — looks up each step by `NodeId` from the (intact) `steps` map and calls `step.step(...)` on the borrow.
- **`backward_retain`** — calls `build_tape_retaining` + `execute_steps_retaining`, skips all cleanup.

```rust
pub fn backward_retain(&mut self, grads: Gradients, node_id: NodeId) -> Gradients {
    let (tape, checkpointer) = self.build_tape_retaining(node_id);
    Self::execute_steps_retaining(tape, &self.steps, grads, checkpointer)
    // No cleanup — graph is preserved
}
```

#### 7. Public API surface

The method was threaded through the full call stack:

| Layer | Addition |
|-------|----------|
| `AutodiffClient` trait | `backward_retain<B>(&self, tensor: &AutodiffTensor<B>) -> Gradients` |
| `GraphMutexClient` | Implements `backward_retain` — no `GraphCleaner::cleanup_orphaned_entries()` call |
| `AutodiffTensor<B>` | `pub fn backward_retain(&self) -> Gradients` |
| `AutodiffBackend` trait (burn-backend) | `fn backward_retain(tensor: &FloatTensor<Self>) -> Self::Gradients` |
| `Autodiff<B, C>` backend | Implements `backward_retain` |
| `Dispatch` backend | Implements `backward_retain` via `as_autodiff().backward_retain()` |
| `Tensor<B, D>` (burn-tensor) | `pub fn backward_retain(&self) -> B::Gradients` |

### Why Cloning Was Not the Answer

The supervisor explicitly advised against solving this with cloning. The insight: **cloning was only needed because ownership was being extracted**. By keeping the steps in the `HashMap` and borrowing them instead, no clone of `Box<dyn Step>` (which is impossible without a `clone_box` method) was ever needed.

The only clones that were added:
- `ops.state.clone()` in `OpsStep::step` — `S: Clone` was already required
- `Arc<dyn Any + Send>` clone in checkpoint builder — Arc refcount bump, not data copy
- `Arc<dyn RetroForward>` clone — already an `Arc`, trivially cheap

No duplicate `backward_retain` / `build_tape_retain` logic beyond what's strictly necessary.

### Testing

Three new tests in `crates/burn-backend-tests/tests/autodiff/retain_graph.rs`:

1. **`should_produce_same_gradients_on_repeated_backward`** — two retain passes yield identical gradients
2. **`should_match_standard_backward_gradients`** — retain gradients match standard backward gradients
3. **`should_allow_multiple_backward_after_retain`** — retain passes followed by a consuming pass all agree

Tests run against both `NoCheckpointing` (default) and `BalancedCheckpointing` strategies.

All 1763 existing tests continue to pass with no regressions.
