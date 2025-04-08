use std::sync::atomic::{AtomicUsize, Ordering};
use dashmap::DashMap;

/// Constant defining the number of `rank` bits in a node represented as a `usize`.
const RANK_BITS: u32 = usize::BITS.ilog2();

/// Constant defining the number of `parent` bits in a node represented as a `usize`.
const PARENT_BITS: u32 = usize::BITS - RANK_BITS;

/// Maximum allowable size of a lock-free union-find data structure.
pub(crate) const MAX_SIZE: usize = usize::MAX >> RANK_BITS;

/// Thread-safe and lock-free implementation of a union-find (also known as disjoint set) data
/// structure.
///
/// This implementation is based on the algorithm presented in
///
/// > "Wait-free Parallel Algorithms for the Union-Find Problem" \
/// > by Richard J. Anderson and Heather Woll.
pub(crate) struct UFRush {
    /// List of nodes in the union-find structure, represented by atomic unsigned integers.
    pub nodes: DashMap<usize, AtomicUsize>,
}

/// Implementation block for the UFRush struct.
#[allow(dead_code)]
impl UFRush {
    /// Creates a new union-find data structure with a specified number of elements.
    ///
    /// # Arguments
    /// * `size` - Number of elements in the union-find structure.
    ///
    /// # Returns
    /// An instance of [`UFRush`].
    ///
    /// # Panics
    /// This method will panic if the `size` exceeds the [`MAX_SIZE`].
    pub fn new() -> Self {
        let nodes : DashMap<usize, AtomicUsize> = DashMap::new();
        Self {nodes}
    }

    /// Returns the total number of elements in the union-find structure.
    ///
    /// # Returns
    /// The total number of elements.
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Determines whether the elements `x` and `y` belong to the same subset.
    ///
    /// # Arguments
    /// * `x` - The first element.
    /// * `y` - The second element.
    ///
    /// # Returns
    /// [`true`] if `x` and `y` belong to the same subset; [`false`] otherwise.
    ///
    /// # Panics
    /// This method will panic if `x` or `y` are out of bounds.
    ///
    /// # Note
    /// The same operation checks whether two elements belong to the same subset. In a sequential
    /// scenario, this operation could be considered redundant, as it can be constructed from a pair
    /// of find operations. However, when it comes to concurrent environments, providing same as a
    /// basic operation is crucial. This is because in such scenarios, the identifiers of subsets
    /// might change dynamically due to concurrent union operations, making it challenging to reliably
    /// determine if a pair of elements belong to the same subset solely based on the outcomes of
    /// individual `find` operations.
    pub fn same(&self, x: usize, y: usize) -> bool {
        loop {
            let x_rep = self.find(x);
            let y_rep = self.find(y);
            if x_rep == y_rep {
                return true;
            }
            let x_node = self.nodes.get(&x_rep).unwrap().load(Ordering::Relaxed);
            if x_rep == parent(x_node) {
                return false;
            }
        }
    }

    /// Finds the representative of the subset that `x` belongs to.
    ///
    /// # Arguments
    /// * `x` - The element to find the representative for.
    ///
    /// # Returns
    /// The representative element of the subset that contains `x`.
    ///
    /// # Panics
    /// This method will panic if `x` is out of bounds.
    ///
    /// # Note
    /// The find operation uses the "path halving" technique, an intermediate strategy between full
    /// path compression and no compression at all.
    ///
    /// In the path halving technique, instead of making every node in the path point directly to the
    /// root as in full path compression, we only change the parent of every other node in the path to
    /// point to its grandparent. This is achieved by skipping over the parent node on each iteration
    /// during the find operation. Despite not fully compressing the path, this strategy is still
    /// effective in flattening the tree structure over time, thus accelerating future operations.
    ///
    /// The advantage of path halving is that it achieves a good balance between the speed of the find
    /// operation and the amount of modification it makes to the tree structure, avoiding a potential
    /// slowdown due to excessively frequent writes in highly concurrent scenarios. Therefore, it is
    /// particularly suitable for lock-free data structures like [`UFRush`], where minimizing
    /// write contention is crucial for performance.
    pub fn find(&self, mut x: usize) -> usize {
        //assert!(x < self.size());
        self.nodes.entry(x).or_insert(AtomicUsize::new(x));
        let mut x_node = self.nodes.get(&x).unwrap().load(Ordering::Relaxed);
        while x != parent(x_node) {
            let x_parent = parent(x_node);
            let x_parent_node = self.nodes.get(&x_parent).unwrap().load(Ordering::Relaxed);
            let x_parent_parent = parent(x_parent_node);

            let x_new_node = encode(x_parent_parent, rank(x_node));
            let _ = self.nodes.get(&x).unwrap().compare_exchange_weak(
                x_node,
                x_new_node,
                Ordering::Release,
                Ordering::Relaxed,
            );

            x = x_parent_parent;
            x_node = self.nodes.get(&x).unwrap().load(Ordering::Relaxed);
        }
        x
    }


    pub fn find_path_compression(&self, mut x: usize) -> usize {
        self.nodes.entry(x).or_insert(AtomicUsize::new(x));

        let mut root = x;
        let mut root_node = self.nodes.get(&root).unwrap().load(Ordering::Relaxed);
        while root != parent(root_node) {
            root = parent(root_node);
            root_node = self.nodes.get(&root).unwrap().load(Ordering::Relaxed);
        }

        while x != root {
            let x_node = self.nodes.get(&x).unwrap().load(Ordering::Relaxed);
            let x_parent = parent(x_node);
            let x_new_node = encode(root, rank(x_node));
            let _ = self.nodes.get(&x).unwrap().compare_exchange_weak(
                x_node, 
                x_new_node,
                Ordering::Release,
                Ordering::Relaxed
            );
            x = x_parent;
        }
        root
    }


    /// Unites the subsets that contain `x` and `y`.
    ///
    /// If `x` and `y` are already in the same subset, no action is performed.
    ///
    /// # Arguments
    /// * `x` - The first element.
    /// * `y` - The second element.
    ///
    /// # Returns
    /// [`true`] if `x` and `y` were in different subsets and a union operation was performed;
    /// [`false`] if `x` and `y` were already in the same subset.
    ///
    /// # Panics
    /// This method will panic if `x` or `y` are out of bounds.
    ///
    /// # Note
    /// The unite operation utilizes a Union-Find algorithm that adopts the "union by rank"
    /// strategy for its union operation.
    ///
    /// In "union by rank", each node holds a rank, and when two sets are united, the set with the
    /// smaller rank becomes a subset of the set with the larger rank. If both sets have the same
    /// rank, either one can become a subset of the other, but the rank of the new root is incremented
    /// by one. This strategy ensures that the tree representing the set does not become excessively
    /// deep, which helps keep the operation's time complexity nearly constant.
    pub fn unite(&self, x: usize, y: usize) -> bool {
        loop {
            // Load representative for x and y
            let mut x_rep = self.find(x);
            let mut y_rep = self.find(y);

            // If they are already part of the same set, return false
            if x_rep == y_rep {
                return false;
            }

            // Load the encoded representation of the representatives
            let x_node = self.nodes.get(&x_rep).unwrap().load(Ordering::Relaxed);
            let y_node = self.nodes.get(&y_rep).unwrap().load(Ordering::Relaxed);

            let mut x_rank = rank(x_node);
            let mut y_rank = rank(y_node);

            // Swap the elements around to always make x the smaller one
            if x_rank > y_rank || (x_rank == y_rank && x_rep > y_rep) {
                std::mem::swap(&mut x_rep, &mut y_rep);
                std::mem::swap(&mut x_rank, &mut y_rank);
            }

            // x_rep is a root
            let cur_value = encode(x_rep, x_rank);
            // assign the new root to be y
            let new_value = encode(y_rep, x_rank);
            // change the value of the smaller subtree root to point to the other one
            if self.nodes.get(&x_rep).unwrap()
                .compare_exchange(cur_value, new_value, Ordering::Release, Ordering::Acquire)
                .is_ok()
            {
                // x_repr now points to y_repr
                // If the subtrees has the same height, increase the rank of the new root
                if x_rank == y_rank {
                    let cur_value = encode(y_rep, y_rank);
                    let new_value = encode(y_rep, y_rank + 1);
                    let _ = self.nodes.get(&y_rep).unwrap().compare_exchange_weak(
                        cur_value,
                        new_value,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                }
                return true;
            }
            // A different thread has already merged modified the value of x_repr -> repeat
        }
    }

    /// Clears the union-find structure, making every element a separate subset.
    pub fn clear(&mut self) {
        self.nodes
            .iter_mut()
            .enumerate()
            .for_each(|(i, node)| node.store(i, Ordering::Relaxed));
    }
}

/// This unsafe implementation indicate that [`UFRush`] can safely be shared
/// across threads (`Sync`).
unsafe impl Sync for UFRush {}

/// This unsafe implementation indicate that [`UFRush`] is safe to transfer
/// the ownership between threads (`Send`).
unsafe impl Send for UFRush {}

/// Encodes the parent node and rank into a single `usize`.
fn encode(parent: usize, rank: usize) -> usize {
    parent | (rank << PARENT_BITS)
}

/// Retrieves the parent node from an encoded `usize`.
pub fn parent(n: usize) -> usize {
    n & MAX_SIZE
}

/// Retrieves the rank from an encoded `usize`.
fn rank(n: usize) -> usize {
    n >> PARENT_BITS
}
