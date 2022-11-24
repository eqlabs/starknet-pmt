mod node;
mod tree;

#[derive(Clone, Debug)]
pub enum Node<F: Felt> {
    Binary(BinaryNode<F>),
    Edge(EdgeNode<F>),
}

#[derive(Clone, Debug)]
pub struct BinaryNode<F: Felt> {
    pub left: F,
    pub right: F,
}

#[derive(Clone, Debug)]
pub struct EdgeNode<F: Felt> {
    pub child: F,
    pub path: bitvec::vec::BitVec<bitvec::order::Msb0, u8>,
}

pub trait Storage<F: Felt> {
    type Error: std::error::Error;

    fn get(&self, hash: &F) -> Result<Node<F>, Self::Error>;
}

pub trait Felt: Clone + std::fmt::Debug {
    fn is_zero(&self) -> bool;

    fn add(&self, value: u8) -> Self;

    fn hash(a: &Self, b: &Self) -> Self;

    fn from_bits(bits: &bitvec::slice::BitSlice<bitvec::order::Msb0, u8>) -> Self;
}

