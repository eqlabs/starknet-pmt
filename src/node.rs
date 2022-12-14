//! Contains constructs for describing the nodes in a Binary Merkle Patricia Tree
//! used by Starknet.
//!
//! For more information about how these Starknet trees are structured, see
//! [`MerkleTree`](super::merkle_tree::MerkleTree).
use crate::Felt;

use std::{cell::RefCell, rc::Rc};

use bitvec::{order::Msb0, prelude::BitVec, slice::BitSlice};

/// A node in a Binary Merkle-Patricia Tree graph.
#[derive(Clone, Debug, PartialEq)]
pub enum Node<F: Felt> {
    /// A node that has not been fetched from storage yet.
    ///
    /// As such, all we know is its hash.
    Unresolved(F),
    /// A branch node with exactly two children.
    Binary(BinaryNode<F>),
    /// Describes a path connecting two other nodes.
    Edge(EdgeNode<F>),
    /// A leaf node that contains a value.
    Leaf(F),
}

/// Describes the [Node::Binary] variant.
#[derive(Clone, Debug, PartialEq)]
pub struct BinaryNode<F: Felt> {
    /// The hash of this node. Is [None] if the node
    /// has not yet been committed.
    pub hash: Option<F>,
    /// The height of this node in the tree.
    pub height: usize,
    /// [Left](Direction::Left) child.
    pub left: Rc<RefCell<Node<F>>>,
    /// [Right](Direction::Right) child.
    pub right: Rc<RefCell<Node<F>>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EdgeNode<F: Felt> {
    /// The hash of this node. Is [None] if the node
    /// has not yet been committed.
    pub hash: Option<F>,
    /// The starting height of this node in the tree.
    pub height: usize,
    /// The path this edge takes.
    pub path: BitVec<Msb0, u8>,
    /// The child of this node.
    pub child: Rc<RefCell<Node<F>>>,
}

/// Describes the direction a child of a [BinaryNode] may have.
///
/// Binary nodes have two children, one left and one right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Left,
    Right,
}

impl Direction {
    /// Inverts the [Direction].
    ///
    /// [Left] becomes [Right], and [Right] becomes [Left].
    ///
    /// [Left]: Direction::Left
    /// [Right]: Direction::Right
    pub fn invert(self) -> Direction {
        match self {
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }
}

impl From<bool> for Direction {
    fn from(tf: bool) -> Self {
        match tf {
            true => Direction::Right,
            false => Direction::Left,
        }
    }
}

impl From<Direction> for bool {
    fn from(direction: Direction) -> Self {
        match direction {
            Direction::Left => false,
            Direction::Right => true,
        }
    }
}

impl<F: Felt> BinaryNode<F> {
    /// Maps the key's bit at the binary node's height to a [Direction].
    ///
    /// This can be used to check which direction the key descibes in the context
    /// of this binary node i.e. which direction the child along the key's path would
    /// take.
    pub fn direction(&self, key: &BitSlice<Msb0, u8>) -> Direction {
        key[self.height].into()
    }

    /// Returns the [Left] or [Right] child.
    ///
    /// [Left]: Direction::Left
    /// [Right]: Direction::Right
    pub fn get_child(&self, direction: Direction) -> Rc<RefCell<Node<F>>> {
        match direction {
            Direction::Left => self.left.clone(),
            Direction::Right => self.right.clone(),
        }
    }

    /// If possible, calculates and sets its own hash value.
    ///
    /// Does nothing if the hash is already [Some].
    ///
    /// If either childs hash is [None], then the hash cannot
    /// be calculated and it will remain [None].
    pub(crate) fn calculate_hash(&mut self) {
        if self.hash.is_some() {
            return;
        }

        let left = self.left.borrow();
        let left = match left.hash() {
            Some(hash) => hash,
            None => unreachable!("subtrees have to be commited first"),
        };

        let right = self.right.borrow();
        let right = match right.hash() {
            Some(hash) => hash,
            None => unreachable!("subtrees have to be commited first"),
        };

        self.hash = Some(F::hash(&left, &right));
    }
}

impl<F: Felt> Node<F> {
    /// Convenience function which sets the inner node's hash to [None], if
    /// applicable.
    ///
    /// Used to indicate that this node has been mutated.
    pub fn mark_dirty(&mut self) {
        match self {
            Node::Binary(inner) => inner.hash = None,
            Node::Edge(inner) => inner.hash = None,
            _ => {}
        }
    }

    pub fn is_binary(&self) -> bool {
        matches!(self, Node::Binary(..))
    }

    pub fn as_binary(&self) -> Option<&BinaryNode<F>> {
        match self {
            Node::Binary(binary) => Some(binary),
            _ => None,
        }
    }

    pub fn as_edge(&self) -> Option<&EdgeNode<F>> {
        match self {
            Node::Edge(edge) => Some(edge),
            _ => None,
        }
    }

    pub fn hash(&self) -> Option<&F> {
        match self {
            Node::Unresolved(hash) => Some(hash),
            Node::Binary(binary) => binary.hash.as_ref(),
            Node::Edge(edge) => edge.hash.as_ref(),
            Node::Leaf(value) => Some(value),
        }
    }
}

impl<F: Felt> EdgeNode<F> {
    /// Returns true if the edge node's path matches the same path given by the key.
    pub fn path_matches(&self, key: &BitSlice<Msb0, u8>) -> bool {
        self.path == key[self.height..self.height + self.path.len()]
    }

    /// Returns the common bit prefix between the edge node's path and the given key.
    ///
    /// This is calculated with the edge's height taken into account.
    pub fn common_path(&self, key: &BitSlice<Msb0, u8>) -> &BitSlice<Msb0, u8> {
        let key_path = key.iter().skip(self.height);
        let common_length = key_path
            .zip(self.path.iter())
            .take_while(|(a, b)| a == b)
            .count();

        &self.path[..common_length]
    }

    /// If possible, calculates and sets its own hash value.
    ///
    /// Does nothing if the hash is already [Some].
    ///
    /// If the child's hash is [None], then the hash cannot
    /// be calculated and it will remain [None].
    pub(crate) fn calculate_hash(&mut self) {
        if self.hash.is_some() {
            return;
        }

        let child = self.child.borrow();
        let child = match child.hash() {
            Some(hash) => hash,
            None => unreachable!("subtree has to be commited before"),
        };

        let path = F::from_bits(&self.path);

        let hash = F::hash(&child, &path).add(self.path.len() as u8);
        self.hash = Some(hash);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::starkhash;

    use bitvec::bitvec;
    use stark_hash::StarkHash;

    mod direction {
        use super::*;
        use Direction::*;

        #[test]
        fn invert() {
            assert_eq!(Left.invert(), Right);
            assert_eq!(Right.invert(), Left);
        }

        #[test]
        fn bool_round_trip() {
            assert_eq!(Direction::from(bool::from(Left)), Left);
            assert_eq!(Direction::from(bool::from(Right)), Right);
        }

        #[test]
        fn right_is_true() {
            assert!(bool::from(Right));
        }

        #[test]
        fn left_is_false() {
            assert!(!bool::from(Left));
        }
    }

    mod binary {
        use super::*;

        #[test]
        fn direction() {
            let uut = BinaryNode {
                hash: None,
                height: 1,
                left: Rc::new(RefCell::new(Node::Leaf(starkhash!("0abc")))),
                right: Rc::new(RefCell::new(Node::Leaf(starkhash!("0def")))),
            };

            let mut zero_key = bitvec![Msb0, u8; 1; 251];
            zero_key.set(1, false);

            let mut one_key = bitvec![Msb0, u8; 0; 251];
            one_key.set(1, true);

            let zero_direction = uut.direction(&zero_key);
            let one_direction = uut.direction(&one_key);

            assert_eq!(zero_direction, Direction::from(false));
            assert_eq!(one_direction, Direction::from(true));
        }

        #[test]
        fn get_child() {
            let left = Rc::new(RefCell::new(Node::Leaf(starkhash!("0abc"))));
            let right = Rc::new(RefCell::new(Node::Leaf(starkhash!("0def"))));

            let uut = BinaryNode {
                hash: None,
                height: 1,
                left: left.clone(),
                right: right.clone(),
            };

            use Direction::*;
            assert_eq!(uut.get_child(Left), left);
            assert_eq!(uut.get_child(Right), right);
        }

        #[test]
        fn hash() {
            // Test data taken from starkware cairo-lang repo:
            // https://github.com/starkware-libs/cairo-lang/blob/fc97bdd8322a7df043c87c371634b26c15ed6cee/src/starkware/starkware_utils/commitment_tree/patricia_tree/nodes_test.py#L14
            //
            // Note that the hash function must be exchanged for `async_stark_hash_func`, otherwise it just uses some other test hash function.
            let expected = StarkHash::from_hex_str(
                "0615bb8d47888d2987ad0c63fc06e9e771930986a4dd8adc55617febfcf3639e",
            )
            .unwrap();
            let left = starkhash!("1234");
            let right = starkhash!("abcd");

            let left = Rc::new(RefCell::new(Node::Unresolved(left)));
            let right = Rc::new(RefCell::new(Node::Unresolved(right)));

            let mut uut = BinaryNode {
                hash: None,
                height: 0,
                left,
                right,
            };

            uut.calculate_hash();

            assert_eq!(uut.hash, Some(expected));
        }
    }

    mod edge {
        use super::*;

        #[test]
        fn hash() {
            // Test data taken from starkware cairo-lang repo:
            // https://github.com/starkware-libs/cairo-lang/blob/fc97bdd8322a7df043c87c371634b26c15ed6cee/src/starkware/starkware_utils/commitment_tree/patricia_tree/nodes_test.py#L38
            //
            // Note that the hash function must be exchanged for `async_stark_hash_func`, otherwise it just uses some other test hash function.
            let expected = StarkHash::from_hex_str(
                "1d937094c09b5f8e26a662d21911871e3cbc6858d55cc49af9848ea6fed4e9",
            )
            .unwrap();
            let child = starkhash!("1234ABCD");
            let child = Rc::new(RefCell::new(Node::Unresolved(child)));
            // Path = 42 in binary.
            let path = bitvec![Msb0, u8; 1, 0, 1, 0, 1, 0];

            let mut uut = EdgeNode {
                hash: None,
                height: 0,
                path,
                child,
            };

            uut.calculate_hash();

            assert_eq!(uut.hash, Some(expected));
        }

        mod path_matches {
            use super::*;

            #[test]
            fn full() {
                let key = starkhash!("0123456789abcdef");
                let child = Rc::new(RefCell::new(Node::Leaf(starkhash!("0abc"))));

                let uut = EdgeNode {
                    hash: None,
                    height: 0,
                    path: key.view_bits().to_bitvec(),
                    child,
                };

                assert!(uut.path_matches(key.view_bits()));
            }

            #[test]
            fn prefix() {
                let key = starkhash!("0123456789abcdef");
                let child = Rc::new(RefCell::new(Node::Leaf(starkhash!("0abc"))));

                let path = key.view_bits()[..45].to_bitvec();

                let uut = EdgeNode {
                    hash: None,
                    height: 0,
                    path,
                    child,
                };

                assert!(uut.path_matches(key.view_bits()));
            }

            #[test]
            fn suffix() {
                let key = starkhash!("0123456789abcdef");
                let child = Rc::new(RefCell::new(Node::Leaf(starkhash!("0abc"))));

                let path = key.view_bits()[50..].to_bitvec();

                let uut = EdgeNode {
                    hash: None,
                    height: 50,
                    path,
                    child,
                };

                assert!(uut.path_matches(key.view_bits()));
            }

            #[test]
            fn middle_slice() {
                let key = starkhash!("0123456789abcdef");
                let child = Rc::new(RefCell::new(Node::Leaf(starkhash!("0abc"))));

                let path = key.view_bits()[230..235].to_bitvec();

                let uut = EdgeNode {
                    hash: None,
                    height: 230,
                    path,
                    child,
                };

                assert!(uut.path_matches(key.view_bits()));
            }
        }
    }
}
