//! Starknet utilises a custom Binary Merkle-Patricia Tree to store and organise its state.
//!
//! From an external perspective the tree is similar to a key-value store, where both key
//! and value are [StarkHashes](StarkHash). The difference is that each tree is immutable,
//! and any mutations result in a new tree with a new root. This mutated variant can then
//! be accessed via the new root, and the old variant via the old root.
//!
//! Trees share common nodes to be efficient. These nodes perform reference counting and
//! will get deleted once all references are gone. State can therefore be tracked over time
//! by mutating the current state, and storing the new root. Old states can be dropped by
//! deleting old roots which are no longer required.
//!
//! #### Tree definition
//!
//! It is important to understand that since all keys are [StarkHashes](StarkHash), this means
//! all paths to a key are equally long - 251 bits.
//!
//! Starknet defines three node types for a tree.
//!
//! `Leaf nodes` which represent an actual value stored.
//!
//! `Edge nodes` which connect two nodes, and __must be__ a maximal subtree (i.e. be as
//! long as possible). This latter condition is important as it strictly defines a tree (i.e. all
//! trees with the same leaves must have the same nodes). The path of an edge node can therefore
//! be many bits long.
//!
//! `Binary nodes` is a branch node with two children, left and right. This represents
//! only a single bit on the path to a leaf.
//!
//! A tree storing a single key-value would consist of two nodes. The root node would be an edge node
//! with a path equal to the key. This edge node is connected to a leaf node storing the value.
//!
//! #### Implementation details
//!
//! We've defined an additional node type, an `Unresolved node`. This is used to
//! represent a node who's hash is known, but has not yet been retrieved from storage (and we therefore
//! have no further details about it).
//!
//! Our implementation is a mix of nodes from persistent storage and any mutations are kept in-memory. It is
//! done this way to allow many mutations to a tree before committing only the final result to storage. This
//! may be confusing since we just said trees are immutable -- but since we are only changing the in-memory
//! tree, the immutable tree still exists in storage. One can therefore think of the in-memory tree as containing
//! the state changes between tree `N` and `N + 1`.
//!
//! The in-memory tree is built using a graph of `Rc<RefCell<Node>>` which is a bit painful.

use bitvec::{prelude::BitSlice, prelude::BitVec, prelude::Msb0};
use std::{cell::RefCell, rc::Rc};

use crate::node::{BinaryNode, Direction, EdgeNode, Node};
use crate::{Felt, Storage};

const TREE_HEIGHT: usize = 251;

/// A Starknet binary Merkle-Patricia tree with a specific root entry-point and storage.
///
/// This is used to update, mutate and access global Starknet state as well as individual contract states.
///
/// For more information on how this functions internally, see [here](super::merkle_tree).
#[derive(Debug, Clone)]
pub struct Tree<F: Felt> {
    root: Option<Rc<RefCell<Node<F>>>>,
}

impl<F: Felt> Tree<F> {
    pub fn empty() -> Self {
        Self { root: None }
    }

    pub fn load(root: F) -> Self {
        if root.is_zero() {
            Self::empty()
        } else {
            Self {
                root: Some(Rc::new(RefCell::new(Node::Unresolved(root)))),
            }
        }
    }

    /// Persists all changes to storage and returns the new root hash.
    ///
    /// Note that the root is reference counted in storage. Committing the
    /// same tree again will therefore increment the count again.
    pub fn commit(&self) -> Vec<(F, crate::Node<F>)> {
        // Go through tree, collect dirty nodes, calculate their hashes and
        // persist them. Take care to increment ref counts of child nodes. So in order
        // to do this correctly, will have to start back-to-front.
        let mut nodes = Vec::new();
        let root = match &self.root {
            Some(root) => root.clone(),
            None => return nodes,
        };
        self.commit_subtree(&mut root.borrow_mut(), &mut nodes);

        nodes
    }

    /// Persists any changes in this subtree to storage.
    ///
    /// This necessitates recursively calculating the hash of, and
    /// in turn persisting, any changed child nodes. This is necessary
    /// as the parent node's hash relies on its childrens hashes.
    ///
    /// In effect, the entire subtree gets persisted.
    fn commit_subtree(&self, subtree_root: &mut Node<F>, mut nodes: &mut Vec<(F, crate::Node<F>)>) {
        use Node::*;
        match subtree_root {
            Unresolved(_) => { /* Unresolved nodes were already committed in the past */ }
            Leaf(_) => { /* Redundant data, and should not be stored */ }
            Binary(binary) if binary.hash.is_some() => { /* not dirty, already committed */ }
            Edge(edge) if edge.hash.is_some() => { /* not dirty, already committed */ }

            Binary(binary) => {
                self.commit_subtree(&mut binary.left.borrow_mut(), &mut nodes);
                self.commit_subtree(&mut binary.right.borrow_mut(), &mut nodes);
                // This will succeed as `commit_subtree` will set the child hashes.
                binary.calculate_hash();
                // unwrap is safe as `commit_subtree` will set the hashes.
                let left = binary.left.borrow().hash().unwrap().clone();
                let right = binary.right.borrow().hash().unwrap().clone();
                let node = crate::Node::Binary(crate::BinaryNode { left, right });

                nodes.push((binary.hash.clone().unwrap(), node));
            }

            Edge(edge) => {
                self.commit_subtree(&mut edge.child.borrow_mut(), &mut nodes);
                // This will succeed as `commit_subtree` will set the child's hash.
                edge.calculate_hash();

                // unwrap is safe as `commit_subtree` will set the hash.
                let child = edge.child.borrow().hash().unwrap().clone();
                let node = crate::Node::Edge(crate::EdgeNode {
                    child,
                    path: edge.path.clone(),
                });

                nodes.push((edge.hash.clone().unwrap(), node));
            }
        }
    }

    /// Sets the value of a key. To delete a key, set the value to [StarkHash::ZERO].
    pub fn set<S: Storage<F>>(
        &mut self,
        storage: &S,
        key: &BitSlice<Msb0, u8>,
        value: F,
    ) -> Result<(), S::Error> {
        if value.is_zero() {
            return self.delete_leaf(storage, key);
        }

        // Changing or inserting a new leaf into the tree will change the hashes
        // of all nodes along the path to the leaf.
        let path = self.traverse(storage, key)?;
        for node in &path {
            node.borrow_mut().mark_dirty();
        }

        // There are three possibilities.
        //
        // 1. The leaf exists, in which case we simply change its value.
        //
        // 2. The tree is empty, we insert the new leaf and the root becomes an edge node connecting to it.
        //
        // 3. The leaf does not exist, and the tree is not empty. The final node in the traversal will
        //    be an edge node who's path diverges from our new leaf node's.
        //
        //    This edge must be split into a new subtree containing both the existing edge's child and the
        //    new leaf. This requires an edge followed by a binary node and then further edges to both the
        //    current child and the new leaf. Any of these new edges may also end with an empty path in
        //    which case they should be elided. It depends on the common path length of the current edge
        //    and the new leaf i.e. the split may be at the first bit (in which case there is no leading
        //    edge), or the split may be in the middle (requires both leading and post edges), or the
        //    split may be the final bit (no post edge).
        use Node::*;
        match path.last() {
            Some(node) => {
                let updated = match &*node.borrow() {
                    Edge(edge) => {
                        let common = edge.common_path(key);

                        // Height of the binary node
                        let branch_height = edge.height + common.len();
                        // Height of the binary node's children
                        let child_height = branch_height + 1;

                        // Path from binary node to new leaf
                        let new_path = key[child_height..].to_bitvec();
                        // Path from binary node to existing child
                        let old_path = edge.path[common.len() + 1..].to_bitvec();

                        // The new leaf branch of the binary node.
                        // (this may be edge -> leaf, or just leaf depending).
                        let new_leaf = Node::Leaf(value);
                        let new = match new_path.is_empty() {
                            true => Rc::new(RefCell::new(new_leaf)),
                            false => {
                                let new_edge = Node::Edge(EdgeNode {
                                    hash: None,
                                    height: child_height,
                                    path: new_path,
                                    child: Rc::new(RefCell::new(new_leaf)),
                                });
                                Rc::new(RefCell::new(new_edge))
                            }
                        };

                        // The existing child branch of the binary node.
                        let old = match old_path.is_empty() {
                            true => edge.child.clone(),
                            false => {
                                let old_edge = Node::Edge(EdgeNode {
                                    hash: None,
                                    height: child_height,
                                    path: old_path,
                                    child: edge.child.clone(),
                                });
                                Rc::new(RefCell::new(old_edge))
                            }
                        };

                        let new_direction = Direction::from(key[branch_height]);
                        let (left, right) = match new_direction {
                            Direction::Left => (new, old),
                            Direction::Right => (old, new),
                        };

                        let branch = Node::Binary(BinaryNode {
                            hash: None,
                            height: branch_height,
                            left,
                            right,
                        });

                        // We may require an edge leading to the binary node.
                        match common.is_empty() {
                            true => branch,
                            false => Node::Edge(EdgeNode {
                                hash: None,
                                height: edge.height,
                                path: common.to_bitvec(),
                                child: Rc::new(RefCell::new(branch)),
                            }),
                        }
                    }
                    // Leaf exists, we replace its value.
                    Leaf(_) => Node::Leaf(value),
                    Unresolved(_) | Binary(_) => {
                        unreachable!("The end of a traversion cannot be unresolved or binary")
                    }
                };

                node.swap(&RefCell::new(updated));
            }
            None => {
                // Getting no travel nodes implies that the tree is empty.
                //
                // Create a new leaf node with the value, and the root becomes
                // an edge node connecting to the leaf.
                let leaf = Node::Leaf(value);
                let edge = Node::Edge(EdgeNode {
                    hash: None,
                    height: 0,
                    path: key.to_bitvec(),
                    child: Rc::new(RefCell::new(leaf)),
                });

                self.root = Some(Rc::new(RefCell::new(edge)));
            }
        }

        Ok(())
    }

    /// Deletes a leaf node from the tree.
    ///
    /// This is not an external facing API; the functionality is instead accessed by calling
    /// [`MerkleTree::set`] with value set to [`StarkHash::ZERO`].
    fn delete_leaf<S: Storage<F>>(
        &mut self,
        storage: &S,
        key: &BitSlice<Msb0, u8>,
    ) -> Result<(), S::Error> {
        // Algorithm explanation:
        //
        // The leaf's parent node is either an edge, or a binary node.
        // If it's an edge node, then it must also be deleted. And its parent
        // must be a binary node. In either case we end up with a binary node
        // who's one child is deleted. This changes the binary to an edge node.
        //
        // Note that its possible that there is no binary node -- if the resulting tree would be empty.
        //
        // This new edge node may need to merge with the old binary node's parent node
        // and other remaining child node -- if they're also edges.
        //
        // Then we are done.
        let path = self.traverse(storage, key)?;

        // Do nothing if the leaf does not exist.
        match path.last() {
            Some(node) => match &*node.borrow() {
                Node::Leaf(_) => {}
                _ => return Ok(()),
            },
            None => return Ok(()),
        }

        // All hashes along the path will become invalid (if they aren't deleted).
        for node in &path {
            node.borrow_mut().mark_dirty();
        }

        // Go backwards until we hit a branch node.
        let mut node_iter = path
            .into_iter()
            .rev()
            .skip_while(|node| !node.borrow().is_binary());

        match node_iter.next() {
            Some(node) => {
                let new_edge = {
                    // This node must be a binary node due to the iteration condition.
                    let binary = node.borrow().as_binary().cloned().unwrap();
                    // Create an edge node to replace the old binary node
                    // i.e. with the remaining child (note the direction invert),
                    //      and a path of just a single bit.
                    let direction = binary.direction(key).invert();
                    let child = binary.get_child(direction);
                    let path = std::iter::once(bool::from(direction)).collect::<BitVec<_, _>>();
                    let mut edge = EdgeNode {
                        hash: None,
                        height: binary.height,
                        path,
                        child,
                    };

                    // Merge the remaining child if it's an edge.
                    self.merge_edges(storage, &mut edge)?;

                    edge
                };
                // Replace the old binary node with the new edge node.
                node.swap(&RefCell::new(Node::Edge(new_edge)));
            }
            None => {
                // We reached the root without a hitting binary node. The new tree
                // must therefore be empty.
                self.root = None;
                return Ok(());
            }
        };

        // Check the parent of the new edge. If it is also an edge, then they must merge.
        if let Some(node) = node_iter.next() {
            if let Node::Edge(edge) = &mut *node.borrow_mut() {
                self.merge_edges(storage, edge)?;
            }
        }

        Ok(())
    }

    /// Returns the value stored at key, or [StarkHash::ZERO] if it does not exist.
    pub fn get<S: Storage<F>>(
        &self,
        storage: &S,
        key: &BitSlice<Msb0, u8>,
    ) -> Result<Option<F>, S::Error> {
        let result = self
            .traverse(storage, key)?
            .last()
            .and_then(|node| match &*node.borrow() {
                Node::Leaf(value) if !value.is_zero() => Some(value.clone()),
                _ => None,
            });
        Ok(result)
    }

    /// Traverses from the current root towards the destination [Leaf](Node::Leaf) node.
    /// Returns the list of nodes along the path.
    ///
    /// If the destination node exists, it will be the final node in the list.
    ///
    /// This means that the final node will always be either a the destination [Leaf](Node::Leaf) node,
    /// or an [Edge](Node::Edge) node who's path suffix does not match the leaf's path.
    ///
    /// The final node can __not__ be a [Binary](Node::Binary) node since it would always be possible to continue
    /// on towards the destination. Nor can it be an [Unresolved](Node::Unresolved) node since this would be
    /// resolved to check if we can travel further.
    fn traverse<S: Storage<F>>(
        &self,
        storage: &S,
        dst: &BitSlice<Msb0, u8>,
    ) -> Result<Vec<Rc<RefCell<Node<F>>>>, S::Error> {
        let mut current = match &self.root {
            None => return Ok(Vec::new()),
            Some(root) => root.clone(),
        };
        let mut height = 0;
        let mut nodes = Vec::new();
        loop {
            use Node::*;

            let current_tmp = current.borrow().clone();

            let next = match current_tmp {
                Unresolved(hash) => {
                    let node = self.resolve(storage, &hash, height)?;
                    current.swap(&RefCell::new(node));
                    current
                }
                Binary(binary) => {
                    nodes.push(current.clone());
                    let next = binary.direction(dst);
                    let next = binary.get_child(next);
                    height += 1;
                    next
                }
                Edge(edge) if edge.path_matches(dst) => {
                    nodes.push(current.clone());
                    height += edge.path.len();
                    edge.child.clone()
                }
                Leaf(_) | Edge(_) => {
                    nodes.push(current);
                    return Ok(nodes);
                }
            };

            current = next;
        }
    }

    /// Retrieves the requested node from storage.
    ///
    /// Result will be either a [Binary](Node::Binary), [Edge](Node::Edge) or [Leaf](Node::Leaf) node.
    fn resolve<S: Storage<F>>(
        &self,
        storage: &S,
        hash: &F,
        height: usize,
    ) -> Result<Node<F>, S::Error> {
        if height == TREE_HEIGHT {
            return Ok(Node::Leaf(hash.clone()));
        }

        let node = storage.get(hash)?;

        let node = match node {
            crate::Node::Binary(binary) => Node::Binary(BinaryNode {
                hash: Some(hash.clone()),
                height,
                left: Rc::new(RefCell::new(Node::Unresolved(binary.left))),
                right: Rc::new(RefCell::new(Node::Unresolved(binary.right))),
            }),
            crate::Node::Edge(edge) => Node::Edge(EdgeNode {
                hash: Some(hash.clone()),
                height,
                path: edge.path,
                child: Rc::new(RefCell::new(Node::Unresolved(edge.child))),
            }),
        };

        Ok(node)
    }

    /// This is a convenience function which merges the edge node with its child __iff__ it is also an edge.
    ///
    /// Does nothing if the child is not also an edge node.
    ///
    /// This can occur when mutating the tree (e.g. deleting a child of a binary node), and is an illegal state
    /// (since edge nodes __must be__ maximal subtrees).
    fn merge_edges<S: Storage<F>>(
        &self,
        storage: &S,
        parent: &mut EdgeNode<F>,
    ) -> Result<(), S::Error> {
        let resolved_child = match &*parent.child.borrow() {
            Node::Unresolved(hash) => {
                self.resolve(storage, hash, parent.height + parent.path.len())?
            }
            other => other.clone(),
        };

        if let Some(child_edge) = resolved_child.as_edge().cloned() {
            parent.path.extend_from_bitslice(&child_edge.path);
            parent.child = child_edge.child;
        }

        Ok(())
    }

    // / Visits all of the nodes in the tree in pre-order using the given visitor function.
    // /
    // / For each node, there will first be a visit for `Node::Unresolved(hash)` followed by visit
    // / at the loaded node when [`Visit::ContinueDeeper`] is returned. At any time the visitor
    // / function can also return `ControlFlow::Break` to stop the visit with the given return
    // / value, which will be returned as `Some(value))` to the caller.
    // /
    // / The visitor function receives the node being visited, as well as the full path to that node.
    // /
    // / Upon successful non-breaking visit of the tree, `None` will be returned.
    // #[allow(dead_code)]
    // pub fn dfs<X, VisitorFn>(&self, visitor_fn: &mut VisitorFn) -> anyhow::Result<Option<X>>
    // where
    //     VisitorFn: FnMut(&Node, &BitSlice<Msb0, u8>) -> ControlFlow<X, Visit>,
    // {
    //     use bitvec::prelude::bitvec;

    //     #[allow(dead_code)]
    //     struct VisitedNode {
    //         node: Rc<RefCell<Node>>,
    //         path: BitVec<Msb0, u8>,
    //     }

    //     let mut visiting = vec![VisitedNode {
    //         node: self.root.clone(),
    //         path: bitvec![Msb0, u8;],
    //     }];

    //     loop {
    //         match visiting.pop() {
    //             None => break,
    //             Some(VisitedNode { node, path }) => {
    //                 let current_node = &*node.borrow();
    //                 if !matches!(current_node, Node::Unresolved(StarkHash::ZERO)) {
    //                     match visitor_fn(current_node, &path) {
    //                         ControlFlow::Continue(Visit::ContinueDeeper) => {
    //                             // the default, no action, just continue deeper
    //                         }
    //                         ControlFlow::Continue(Visit::StopSubtree) => {
    //                             // make sure we don't add any more to `visiting` on this subtree
    //                             continue;
    //                         }
    //                         ControlFlow::Break(x) => {
    //                             // early exit
    //                             return Ok(Some(x));
    //                         }
    //                     }
    //                 }
    //                 match current_node {
    //                     Node::Binary(b) => {
    //                         visiting.push(VisitedNode {
    //                             node: b.right.clone(),
    //                             path: {
    //                                 let mut path_right = path.clone();
    //                                 path_right.push(Direction::Right.into());
    //                                 path_right
    //                             },
    //                         });
    //                         visiting.push(VisitedNode {
    //                             node: b.left.clone(),
    //                             path: {
    //                                 let mut path_left = path.clone();
    //                                 path_left.push(Direction::Left.into());
    //                                 path_left
    //                             },
    //                         });
    //                     }
    //                     Node::Edge(e) => {
    //                         visiting.push(VisitedNode {
    //                             node: e.child.clone(),
    //                             path: {
    //                                 let mut extended_path = path.clone();
    //                                 extended_path.extend_from_bitslice(&e.path);
    //                                 extended_path
    //                             },
    //                         });
    //                     }
    //                     Node::Leaf(_) => {}
    //                     Node::Unresolved(hash) => {
    //                         // Zero means empty tree, so nothing to resolve
    //                         if hash != &StarkHash::ZERO {
    //                             visiting.push(VisitedNode {
    //                                 node: Rc::new(RefCell::new(self.resolve(*hash, path.len())?)),
    //                                 path,
    //                             });
    //                         }
    //                     }
    //                 };
    //             }
    //         }
    //     }

    //     Ok(None)
    // }
}

/// Direction for the [`MerkleTree::dfs`] as the return value of the visitor function.
// #[derive(Default)]
// pub enum Visit {
//     /// Instructs that the visit should visit any subtrees of the current node. This is a no-op for
//     /// [`Node::Leaf`].
//     #[default]
//     ContinueDeeper,
//     /// Returning this value for [`Node::Binary`] or [`Node::Edge`] will ignore all of the children
//     /// of the node for the rest of the iteration. This is useful because two trees often share a
//     /// number of subtrees with earlier blocks. Returning this for [`Node::Leaf`] is a no-op.
//     StopSubtree,
// }

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::starkhash;
    use bitvec::prelude::*;
    use stark_hash::StarkHash;

    #[derive(Debug)]
    pub struct HashMapStorageError;
    impl std::fmt::Display for HashMapStorageError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("node data not found")
        }
    }
    impl std::error::Error for HashMapStorageError {}

    impl<F: Felt + Eq + std::hash::Hash + PartialEq> Storage<F> for HashMap<F, crate::Node<F>> {
        type Error = HashMapStorageError;

        fn get(&self, hash: &F) -> Result<crate::Node<F>, Self::Error> {
            self.get(hash).ok_or(HashMapStorageError).cloned()
        }
    }

    #[test]
    fn get_empty() {
        let storage = HashMap::new();
        let uut = Tree::<StarkHash>::empty();

        let key = starkhash!("99cadc82").view_bits().to_bitvec();
        assert_eq!(uut.get(&storage, &key).unwrap(), None);
    }

    mod set {
        use super::*;

        #[test]
        fn set_get() {
            let storage = HashMap::new();
            let mut uut = Tree::<StarkHash>::empty();

            let key0 = starkhash!("99cadc82").view_bits().to_bitvec();
            let key1 = starkhash!("901823").view_bits().to_bitvec();
            let key2 = starkhash!("8975").view_bits().to_bitvec();

            let val0 = starkhash!("891127cbaf");
            let val1 = starkhash!("82233127cbaf");
            let val2 = starkhash!("0891124667aacde7cbaf");

            uut.set(&storage, &key0, val0).unwrap();
            uut.set(&storage, &key1, val1).unwrap();
            uut.set(&storage, &key2, val2).unwrap();

            assert_eq!(uut.get(&storage, &key0).unwrap(), Some(val0));
            assert_eq!(uut.get(&storage, &key1).unwrap(), Some(val1));
            assert_eq!(uut.get(&storage, &key2).unwrap(), Some(val2));
        }

        #[test]
        fn overwrite() {
            let storage = HashMap::new();
            let mut uut = Tree::<StarkHash>::empty();

            let key = starkhash!("0123").view_bits().to_bitvec();
            let old_value = starkhash!("0abc");
            let new_value = starkhash!("0def");

            uut.set(&storage, &key, old_value).unwrap();
            uut.set(&storage, &key, new_value).unwrap();

            assert_eq!(uut.get(&storage, &key).unwrap(), Some(new_value));
        }
    }

    mod tree_state {
        use super::*;

        #[test]
        fn single_leaf() {
            let storage = HashMap::new();
            let mut uut = Tree::<StarkHash>::empty();

            let key = starkhash!("0123").view_bits().to_bitvec();
            let value = starkhash!("0abc");

            uut.set(&storage, &key, value).unwrap();

            // The tree should consist of an edge node (root) leading to a leaf node.
            // The edge node path should match the key, and the leaf node the value.
            let expected_path = key.clone();

            let edge = uut
                .root
                .unwrap()
                .borrow()
                .as_edge()
                .cloned()
                .expect("root should be an edge");
            assert_eq!(edge.path, expected_path);
            assert_eq!(edge.height, 0);

            let leaf = edge.child.borrow().to_owned();
            assert_eq!(leaf, Node::Leaf(value));
        }

        #[test]
        fn binary_middle() {
            let key0 = bitvec![Msb0, u8; 0; 251];

            let mut key1 = bitvec![Msb0, u8; 0; 251];
            key1.set(50, true);

            let value0 = starkhash!("0abc");
            let value1 = starkhash!("0def");

            let storage = HashMap::new();
            let mut uut = Tree::<StarkHash>::empty();

            uut.set(&storage, &key0, value0).unwrap();
            uut.set(&storage, &key1, value1).unwrap();

            let edge = uut
                .root
                .unwrap()
                .borrow()
                .as_edge()
                .cloned()
                .expect("root should be an edge");

            let expected_path = bitvec![Msb0, u8; 0; 50];
            assert_eq!(edge.path, expected_path);
            assert_eq!(edge.height, 0);

            let binary = edge
                .child
                .borrow()
                .as_binary()
                .cloned()
                .expect("should be a binary node");

            assert_eq!(binary.height, 50);

            let direction0 = Direction::from(false);
            let direction1 = Direction::from(true);

            let child0 = binary
                .get_child(direction0)
                .borrow()
                .as_edge()
                .cloned()
                .expect("child should be an edge");
            let child1 = binary
                .get_child(direction1)
                .borrow()
                .as_edge()
                .cloned()
                .expect("child should be an edge");

            assert_eq!(child0.height, 51);
            assert_eq!(child1.height, 51);

            let leaf0 = child0.child.borrow().to_owned();
            let leaf1 = child1.child.borrow().to_owned();

            assert_eq!(leaf0, Node::Leaf(value0));
            assert_eq!(leaf1, Node::Leaf(value1));
        }

        #[test]
        fn binary_root() {
            let key0 = bitvec![Msb0, u8; 0; 251];

            let mut key1 = bitvec![Msb0, u8; 0; 251];
            key1.set(0, true);

            let value0 = starkhash!("0abc");
            let value1 = starkhash!("0def");

            let storage = HashMap::new();
            let mut uut = Tree::<StarkHash>::empty();

            uut.set(&storage, &key0, value0).unwrap();
            uut.set(&storage, &key1, value1).unwrap();

            let binary = uut
                .root
                .unwrap()
                .borrow()
                .as_binary()
                .cloned()
                .expect("root should be a binary node");

            assert_eq!(binary.height, 0);

            let direction0 = Direction::from(false);
            let direction1 = Direction::from(true);

            let child0 = binary
                .get_child(direction0)
                .borrow()
                .as_edge()
                .cloned()
                .expect("child should be an edge");
            let child1 = binary
                .get_child(direction1)
                .borrow()
                .as_edge()
                .cloned()
                .expect("child should be an edge");

            assert_eq!(child0.height, 1);
            assert_eq!(child1.height, 1);

            let leaf0 = child0.child.borrow().to_owned();
            let leaf1 = child1.child.borrow().to_owned();

            assert_eq!(leaf0, Node::Leaf(value0));
            assert_eq!(leaf1, Node::Leaf(value1));
        }

        #[test]
        fn binary_leaves() {
            let key0 = starkhash!("00").view_bits().to_bitvec();
            let key1 = starkhash!("01").view_bits().to_bitvec();
            let value0 = starkhash!("0abc");
            let value1 = starkhash!("0def");

            let storage = HashMap::new();
            let mut uut = Tree::<StarkHash>::empty();

            uut.set(&storage, &key0, value0).unwrap();
            uut.set(&storage, &key1, value1).unwrap();

            // The tree should consist of an edge node, terminating in a binary node connecting to
            // the two leaf nodes.
            let edge = uut
                .root
                .unwrap()
                .borrow()
                .as_edge()
                .cloned()
                .expect("root should be an edge");
            // The edge's path will be the full key path excluding the final bit.
            // The final bit is represented by the following binary node.
            let mut expected_path = key0.to_bitvec();
            expected_path.pop();

            assert_eq!(edge.path, expected_path);
            assert_eq!(edge.height, 0);

            let binary = edge
                .child
                .borrow()
                .as_binary()
                .cloned()
                .expect("should be a binary node");
            assert_eq!(binary.height, 250);

            // The binary children should be the leaf nodes.
            let direction0 = Direction::from(false);
            let direction1 = Direction::from(true);
            let child0 = binary.get_child(direction0).borrow().to_owned();
            let child1 = binary.get_child(direction1).borrow().to_owned();
            assert_eq!(child0, Node::Leaf(value0));
            assert_eq!(child1, Node::Leaf(value1));
        }
    }

    mod delete_leaf {
        use super::*;

        #[test]
        fn empty() {
            let storage = HashMap::new();
            let mut uut = Tree::<StarkHash>::empty();

            let key = starkhash!("123abc").view_bits().to_bitvec();
            uut.delete_leaf(&storage, &key).unwrap();

            assert!(uut.root.is_none());
        }

        #[test]
        fn single_insert_and_removal() {
            let storage = HashMap::new();
            let mut uut = Tree::<StarkHash>::empty();

            let key = starkhash!("0123").view_bits().to_bitvec();
            let value = starkhash!("0abc");

            uut.set(&storage, &key, value).unwrap();
            uut.delete_leaf(&storage, &key).unwrap();

            assert_eq!(uut.get(&storage, &key).unwrap(), None);
            assert!(uut.root.is_none());
        }

        #[test]
        fn three_leaves_and_one_removal() {
            let storage = HashMap::new();
            let mut uut = Tree::<StarkHash>::empty();

            let key0 = starkhash!("99cadc82").view_bits().to_bitvec();
            let key1 = starkhash!("901823").view_bits().to_bitvec();
            let key2 = starkhash!("8975").view_bits().to_bitvec();

            let val0 = starkhash!("01");
            let val1 = starkhash!("02");
            let val2 = starkhash!("03");

            uut.set(&storage, &key0, val0).unwrap();
            uut.set(&storage, &key1, val1).unwrap();
            uut.set(&storage, &key2, val2).unwrap();

            uut.delete_leaf(&storage, &key1).unwrap();

            assert_eq!(uut.get(&storage, &key0).unwrap(), Some(val0));
            assert_eq!(uut.get(&storage, &key1).unwrap(), None);
            assert_eq!(uut.get(&storage, &key2).unwrap(), Some(val2));
        }
    }

    mod persistence {
        use super::*;

        #[test]
        fn set() {
            let mut storage = HashMap::new();
            let mut uut = Tree::<StarkHash>::empty();

            let key0 = starkhash!("99cadc82").view_bits().to_bitvec();
            let key1 = starkhash!("901823").view_bits().to_bitvec();
            let key2 = starkhash!("8975").view_bits().to_bitvec();

            let val0 = starkhash!("01");
            let val1 = starkhash!("02");
            let val2 = starkhash!("03");

            uut.set(&storage, &key0, val0).unwrap();
            uut.set(&storage, &key1, val1).unwrap();
            uut.set(&storage, &key2, val2).unwrap();

            let new_nodes = uut.commit();
            let root = new_nodes.last().unwrap().0;
            for (hash, node) in new_nodes {
                storage.insert(hash, node);
            }

            let uut = Tree::load(root);

            assert_eq!(uut.get(&storage, &key0).unwrap(), Some(val0));
            assert_eq!(uut.get(&storage, &key1).unwrap(), Some(val1));
            assert_eq!(uut.get(&storage, &key2).unwrap(), Some(val2));
        }

        #[test]
        fn delete_leaf_regression() {
            // This test exercises a bug in the merging of edge nodes. It was caused
            // by the merge code not resolving unresolved nodes. This meant that
            // unresolved edge nodes would not get merged with the parent edge node
            // causing a malformed tree.
            let mut storage = HashMap::new();
            let mut uut = Tree::<StarkHash>::empty();

            let leaves = [
                (
                    starkhash!("01A2FD9B06EAB5BCA4D3885EE4C42736E835A57399FF8B7F6083A92FD2A20095"),
                    starkhash!("0215AA555E0CE3E462423D18B7216378D3CCD5D94D724AC7897FBC83FAAA4ED4"),
                ),
                (
                    starkhash!("07AC69285B869DC3E8B305C748A0B867B2DE3027AECEBA51158ECA3B7354D76F"),
                    starkhash!("065C85592F29501D97A2EA1CCF2BA867E6A838D602F4E7A7391EFCBF66958386"),
                ),
                (
                    starkhash!("05C71AB5EF6A5E9DBC7EFD5C61554AB36039F60E5BA076833102E24344524566"),
                    starkhash!("060970DF8E8A19AF3F41B78E93B845EC074A0AED4E96D18C6633580722B93A28"),
                ),
                (
                    starkhash!("0000000000000000000000000000000000000000000000000000000000000005"),
                    starkhash!("000000000000000000000000000000000000000000000000000000000000022B"),
                ),
                (
                    starkhash!("0000000000000000000000000000000000000000000000000000000000000005"),
                    starkhash!("0000000000000000000000000000000000000000000000000000000000000000"),
                ),
            ];

            // Add the first four leaves and commit them to storage.
            for (key, val) in &leaves[..4] {
                let key = key.view_bits();
                uut.set(&storage, key, *val).unwrap();
            }
            let new_nodes = uut.commit();
            let root = new_nodes.last().unwrap().0;
            for (hash, node) in new_nodes {
                storage.insert(hash, node);
            }

            // Delete the final leaf; this exercises the bug as the nodes are all in storage (unresolved).
            let mut uut = Tree::load(root);
            let key = leaves[4].0.view_bits().to_bitvec();
            let val = leaves[4].1;
            uut.set(&storage, &key, val).unwrap();
            let root = uut.commit().last().unwrap().0;
            let expect =
                starkhash!("05f3b2b98faef39c60dbbb459dbe63d1d10f1688af47fbc032f2cab025def896");
            assert_eq!(root, expect);
        }

        mod consecutive_roots {
            use super::*;

            #[test]
            fn set_get() {
                let mut storage = HashMap::new();
                let mut uut = Tree::empty();

                let key0 = starkhash!("99cadc82").view_bits().to_bitvec();
                let key1 = starkhash!("901823").view_bits().to_bitvec();
                let key2 = starkhash!("8975").view_bits().to_bitvec();

                let val0 = starkhash!("01");
                let val1 = starkhash!("02");
                let val2 = starkhash!("03");

                uut.set(&storage, &key0, val0).unwrap();
                let new_nodes = uut.commit();
                let root0 = new_nodes.last().unwrap().0;
                for (hash, node) in new_nodes {
                    storage.insert(hash, node);
                }

                let mut uut = Tree::load(root0);
                uut.set(&storage, &key1, val1).unwrap();
                let new_nodes = uut.commit();
                let root1 = new_nodes.last().unwrap().0;
                for (hash, node) in new_nodes {
                    storage.insert(hash, node);
                }

                let mut uut = Tree::load(root1);
                uut.set(&storage, &key2, val2).unwrap();
                let new_nodes = uut.commit();
                let root2 = new_nodes.last().unwrap().0;
                for (hash, node) in new_nodes {
                    storage.insert(hash, node);
                }

                let uut = Tree::load(root0);
                assert_eq!(uut.get(&storage, &key0).unwrap(), Some(val0));
                assert_eq!(uut.get(&storage, &key1).unwrap(), None);
                assert_eq!(uut.get(&storage, &key2).unwrap(), None);

                let uut = Tree::load(root1);
                assert_eq!(uut.get(&storage, &key0).unwrap(), Some(val0));
                assert_eq!(uut.get(&storage, &key1).unwrap(), Some(val1));
                assert_eq!(uut.get(&storage, &key2).unwrap(), None);

                let uut = Tree::load(root2);
                assert_eq!(uut.get(&storage, &key0).unwrap(), Some(val0));
                assert_eq!(uut.get(&storage, &key1).unwrap(), Some(val1));
                assert_eq!(uut.get(&storage, &key2).unwrap(), Some(val2));
            }
        }

        mod parallel_roots {
            use super::*;

            #[test]
            fn set_get() {
                let mut storage = HashMap::new();
                let mut uut = Tree::empty();

                let key0 = starkhash!("99cadc82").view_bits().to_bitvec();
                let key1 = starkhash!("901823").view_bits().to_bitvec();
                let key2 = starkhash!("8975").view_bits().to_bitvec();

                let val0 = starkhash!("01");
                let val1 = starkhash!("02");
                let val2 = starkhash!("03");

                uut.set(&storage, &key0, val0).unwrap();
                let new_nodes = uut.commit();
                let root0 = new_nodes.last().unwrap().0;
                for (hash, node) in new_nodes {
                    storage.insert(hash, node);
                }

                let mut uut = Tree::load(root0);
                uut.set(&storage, &key1, val1).unwrap();
                let new_nodes = uut.commit();
                let root1 = new_nodes.last().unwrap().0;
                for (hash, node) in new_nodes {
                    storage.insert(hash, node);
                }

                let mut uut = Tree::load(root0);
                uut.set(&storage, &key2, val2).unwrap();
                let new_nodes = uut.commit();
                let root2 = new_nodes.last().unwrap().0;
                for (hash, node) in new_nodes {
                    storage.insert(hash, node);
                }

                let uut = Tree::load(root0);
                assert_eq!(uut.get(&storage, &key0).unwrap(), Some(val0));
                assert_eq!(uut.get(&storage, &key1).unwrap(), None);
                assert_eq!(uut.get(&storage, &key2).unwrap(), None);

                let uut = Tree::load(root1);
                assert_eq!(uut.get(&storage, &key0).unwrap(), Some(val0));
                assert_eq!(uut.get(&storage, &key1).unwrap(), Some(val1));
                assert_eq!(uut.get(&storage, &key2).unwrap(), None);

                let uut = Tree::load(root2);
                assert_eq!(uut.get(&storage, &key0).unwrap(), Some(val0));
                assert_eq!(uut.get(&storage, &key1).unwrap(), None);
                assert_eq!(uut.get(&storage, &key2).unwrap(), Some(val2));
            }
        }
    }

    mod real_world {
        use super::*;
        use crate::starkhash;

        #[test]
        fn simple() {
            // Test data created from Starknet cairo wrangling.
            let storage = HashMap::new();
            let mut uut = Tree::<StarkHash>::empty();

            uut.set(&storage, starkhash!("01").view_bits(), starkhash!("00"))
                .unwrap();

            uut.set(&storage, starkhash!("86").view_bits(), starkhash!("01"))
                .unwrap();

            uut.set(&storage, starkhash!("87").view_bits(), starkhash!("02"))
                .unwrap();

            let root = uut.commit().last().unwrap().0;

            assert_eq!(
                root,
                starkhash!("05458b9f8491e7c845bffa4cd36cdb3a7c29dcdf75f2809bd6f4ce65386facfc")
            );
        }

        #[test]
        fn contract_edge_branches_correctly_on_insert() {
            // This emulates the contract update which exposed a bug in `set`.
            //
            // This was discovered by comparing the global state tree for the
            // gensis block on goerli testnet (alpha 4.0).
            //
            // The bug was identified by comparing root and nodes against the python
            // utility in `root/py/src/test_generate_test_storage_tree.py`.
            let leaves = [
                (starkhash!("05"), starkhash!("66")),
                (
                    starkhash!("01BF95D4B58F0741FEA29F94EE5A118D0847C8B7AE0173C2A570C9F74CCA9EA1"),
                    starkhash!("07E5"),
                ),
                (
                    starkhash!("03C75C20765D020B0EC41B48BB8C5338AC4B619FC950D59994E844E1E1B9D2A9"),
                    starkhash!("07C7"),
                ),
                (
                    starkhash!("04065B936C56F5908A981084DAFA66DC17600937DC80C52EEB834693BB811792"),
                    starkhash!("07970C532B764BB36FAF5696B8BC1317505B8A4DC9EEE5DF4994671757975E4D"),
                ),
                (
                    starkhash!("04B5FBB4904167E2E8195C35F7D4E78501A3FE95896794367C85B60B39AEFFC2"),
                    starkhash!("0232C969EAFC5B30C20648759D7FA1E2F4256AC6604E1921578101DCE4DFDF48"),
                ),
            ];

            // create test database
            let storage = HashMap::new();
            let mut uut = Tree::empty();

            for (key, val) in leaves {
                let key = key.view_bits();
                uut.set(&storage, key, val).unwrap();
            }

            let root = uut.commit().last().unwrap().0;

            let expected =
                starkhash!("06ee9a8202b40f3f76f1a132f953faa2df78b3b33ccb2b4406431abdc99c2dfe");

            assert_eq!(root, expected);
        }
    }

    // mod dfs {
    //     use super::{BinaryNode, EdgeNode, Node, Tree, Visit};
    //     use crate::starkhash;
    //     use bitvec::slice::BitSlice;
    //     use bitvec::{bitvec, prelude::Msb0};
    //     use stark_hash::StarkHash;
    //     use std::cell::RefCell;
    //     use std::ops::ControlFlow;
    //     use std::rc::Rc;

    //     #[test]
    //     fn empty_tree() {
    //         let mut conn = rusqlite::Connection::open_in_memory().unwrap();
    //         let transaction = conn.transaction().unwrap();
    //         let uut = Tree::empty();

    //         let mut visited = vec![];
    //         let mut visitor_fn = |node: &Node, path: &BitSlice<Msb0, u8>| {
    //             visited.push((node.clone(), path.to_bitvec()));
    //             ControlFlow::Continue::<(), Visit>(Default::default())
    //         };
    //         uut.dfs(&mut visitor_fn).unwrap();
    //         assert!(visited.is_empty());
    //     }

    //     #[test]
    //     fn one_leaf() {
    //         let storage = HashMap::new();
    //         let uut = Tree::<StarkHash>::empty();

    //         let key = starkhash!("01");
    //         let value = starkhash!("02");

    //         uut.set(&storage, key.view_bits(), value).unwrap();

    //         let mut visited = vec![];
    //         let mut visitor_fn = |node: &Node, path: &BitSlice<Msb0, u8>| {
    //             visited.push((node.clone(), path.to_bitvec()));
    //             ControlFlow::Continue::<(), Visit>(Default::default())
    //         };
    //         uut.dfs(&mut visitor_fn).unwrap();

    //         assert_eq!(
    //             visited,
    //             vec![
    //                 (
    //                     Node::Edge(EdgeNode {
    //                         hash: None,
    //                         height: 0,
    //                         path: key.view_bits().into(),
    //                         child: Rc::new(RefCell::new(Node::Leaf(value)))
    //                     }),
    //                     bitvec![Msb0, u8;]
    //                 ),
    //                 (Node::Leaf(value), key.view_bits().into())
    //             ],
    //         );
    //     }

    //     #[test]
    //     fn two_leaves() {
    //         let storage = HashMap::new();
    //         let uut = Tree::<StarkHash>::empty();

    //         let key_left = starkhash!("00");
    //         let value_left = starkhash!("02");
    //         let key_right = starkhash!("01");
    //         let value_right = starkhash!("03");

    //         uut.set(&storage, key_right.view_bits(), value_right)
    //             .unwrap();
    //         uut.set(&storage, key_left.view_bits(), value_left).unwrap();

    //         let mut visited = vec![];
    //         let mut visitor_fn = |node: &Node, path: &BitSlice<Msb0, u8>| {
    //             visited.push((node.clone(), path.to_bitvec()));
    //             ControlFlow::Continue::<(), Visit>(Default::default())
    //         };
    //         uut.dfs(&mut visitor_fn).unwrap();

    //         let expected_3 = (Node::Leaf(value_right), key_right.view_bits().into());
    //         let expected_2 = (Node::Leaf(value_left), key_left.view_bits().into());
    //         let expected_1 = (
    //             Node::Binary(BinaryNode {
    //                 hash: None,
    //                 height: 250,
    //                 left: Rc::new(RefCell::new(expected_2.0.clone())),
    //                 right: Rc::new(RefCell::new(expected_3.0.clone())),
    //             }),
    //             bitvec![Msb0, u8; 0; 250],
    //         );
    //         let expected_0 = (
    //             Node::Edge(EdgeNode {
    //                 hash: None,
    //                 height: 0,
    //                 path: bitvec![Msb0, u8; 0; 250],
    //                 child: Rc::new(RefCell::new(expected_1.0.clone())),
    //             }),
    //             bitvec![Msb0, u8;],
    //         );

    //         pretty_assertions::assert_eq!(
    //             visited,
    //             vec![expected_0, expected_1, expected_2, expected_3]
    //         );
    //     }

    //     #[test]
    //     fn three_leaves() {
    //         let storage = HashMap::new();
    //         let uut = Tree::<StarkHash>::empty();

    //         let key_a = starkhash!("10");
    //         let value_a = starkhash!("0a");
    //         let key_b = starkhash!("11");
    //         let value_b = starkhash!("0b");
    //         let key_c = starkhash!("13");
    //         let value_c = starkhash!("0c");

    //         uut.set(&storage, key_c.view_bits(), value_c).unwrap();
    //         uut.set(&storage, key_a.view_bits(), value_a).unwrap();
    //         uut.set(&storage, key_b.view_bits(), value_b).unwrap();

    //         let mut visited = vec![];
    //         let mut visitor_fn = |node: &Node, path: &BitSlice<Msb0, u8>| {
    //             visited.push((node.clone(), path.to_bitvec()));
    //             ControlFlow::Continue::<(), Visit>(Default::default())
    //         };
    //         uut.dfs(&mut visitor_fn).unwrap();

    //         // 0
    //         // |
    //         // 1
    //         // |\
    //         // 2 5
    //         // |\ \
    //         // 3 4 6
    //         // a b c

    //         let path_to_0 = bitvec![Msb0, u8;];
    //         let path_to_1 = {
    //             let mut p = bitvec![Msb0, u8; 0; 249];
    //             *p.get_mut(246).unwrap() = true;
    //             p
    //         };
    //         let mut path_to_2 = path_to_1.clone();
    //         path_to_2.push(false);
    //         let mut path_to_5 = path_to_1.clone();
    //         path_to_5.push(true);

    //         let expected_6 = (Node::Leaf(value_c), key_c.view_bits().into());
    //         let expected_5 = (
    //             Node::Edge(EdgeNode {
    //                 hash: None,
    //                 height: 250,
    //                 path: bitvec![Msb0, u8; 1; 1],
    //                 child: Rc::new(RefCell::new(expected_6.0.clone())),
    //             }),
    //             path_to_5,
    //         );
    //         let expected_4 = (Node::Leaf(value_b), key_b.view_bits().into());
    //         let expected_3 = (Node::Leaf(value_a), key_a.view_bits().into());
    //         let expected_2 = (
    //             Node::Binary(BinaryNode {
    //                 hash: None,
    //                 height: 250,
    //                 left: Rc::new(RefCell::new(expected_3.0.clone())),
    //                 right: Rc::new(RefCell::new(expected_4.0.clone())),
    //             }),
    //             path_to_2,
    //         );
    //         let expected_1 = (
    //             Node::Binary(BinaryNode {
    //                 hash: None,
    //                 height: 249,
    //                 left: Rc::new(RefCell::new(expected_2.0.clone())),
    //                 right: Rc::new(RefCell::new(expected_5.0.clone())),
    //             }),
    //             path_to_1.clone(),
    //         );
    //         let expected_0 = (
    //             Node::Edge(EdgeNode {
    //                 hash: None,
    //                 height: 0,
    //                 path: path_to_1,
    //                 child: Rc::new(RefCell::new(expected_1.0.clone())),
    //             }),
    //             path_to_0,
    //         );

    //         pretty_assertions::assert_eq!(
    //             visited,
    //             vec![
    //                 expected_0, expected_1, expected_2, expected_3, expected_4, expected_5,
    //                 expected_6
    //             ]
    //         );
    //     }
    // }

    // #[test]
    // fn dfs_on_leaf_to_binary_collision_tree() {
    //     let mut conn = rusqlite::Connection::open_in_memory().unwrap();
    //     let transaction = conn.transaction().unwrap();
    //     let mut uut = Tree::empty();

    //     let value = starkhash!("01");
    //     let key0 = starkhash!("ee00").view_bits().to_bitvec();
    //     let key1 = starkhash!("ee01").view_bits().to_bitvec();

    //     let key2 = starkhash!("ffff").view_bits().to_bitvec();
    //     let hash_of_values = stark_hash::stark_hash(value, value);
    //     uut.set(&storage, &key2, hash_of_values).unwrap();

    //     uut.set(&storage, &key0, value).unwrap();
    //     uut.set(&storage, &key1, value).unwrap();

    //     let root = uut.commit().last().unwrap().0;

    //     let uut = Tree::load(root).unwrap();
    //     // this used to panic because it did find the binary on dev profile with the leaf hash
    //     let mut visited = Vec::new();
    //     uut.dfs(&mut |n: &_, p: &_| -> ControlFlow<(), Visit> {
    //         if let Node::Leaf(h) = n {
    //             visited.push((StarkHash::from_bits(p).unwrap(), *h));
    //         }
    //         std::ops::ControlFlow::Continue(Default::default())
    //     })
    //     .unwrap();
    //     assert_eq!(uut.get(&storage, &key0).unwrap(), Some(value));
    //     assert_eq!(uut.get(&storage, &key1).unwrap(), Some(value));
    //     assert_eq!(uut.get(&storage, &key2).unwrap(), Some(hash_of_values));

    //     assert_eq!(
    //         visited,
    //         &[
    //             (starkhash!("EE00"), starkhash!("01")),
    //             (starkhash!("EE01"), starkhash!("01")),
    //             (
    //                 starkhash!("FFFF"),
    //                 starkhash!("02EBBD6878F81E49560AE863BD4EF327A417037BF57B63A016130AD0A94C8EAC")
    //             )
    //         ]
    //     );
    // }
}
