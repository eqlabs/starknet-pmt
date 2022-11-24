mod node;
mod tree;

pub use tree::Tree;

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

#[cfg(test)]
impl Felt for stark_hash::StarkHash {
    fn is_zero(&self) -> bool {
        self.is_zero()
    }

    fn from_bits(bits: &bitvec::slice::BitSlice<bitvec::order::Msb0, u8>) -> Self {
        stark_hash::StarkHash::from_bits(bits)
            .expect("Merkle tree guarantees that bit vector won't exceed 251 bits")
    }

    fn add(&self, value: u8) -> Self {
        let mut buffer = [0; 32];
        buffer[31] = value;
        let value = stark_hash::StarkHash::from_be_bytes(buffer).unwrap();
        *self + value
    }

    fn hash(a: &Self, b: &Self) -> Self {
        stark_hash::stark_hash(*a, *b)
    }
}

#[cfg(test)]
macro_rules! starkhash {
    ($hex:expr) => {{
        let bytes = hex_literal::hex!($hex);
        match stark_hash::StarkHash::from_be_slice(bytes.as_slice()) {
            Ok(sh) => sh,
            Err(stark_hash::OverflowError) => panic!("Invalid constant: OverflowError"),
        }
    }};
}
#[cfg(test)]
pub(crate) use starkhash;
