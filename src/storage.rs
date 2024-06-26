/*
SmartER typing for storing minhash cliques.
For each band we want to store a dict:

signature -> [(path_id, line_num), ...]


With the specs:
signature: needs to be big enough to not allow collisions.
		   If E[collisions] is kept < 1, this means 1
		   Dynamically decided based on expected number of documents.
		   If # pairs is (#documents-choose-2), need number of bins to be greater than 
		   log_2(#documents-choose-2) 
		  
path_id: identifier for which path we are using
	defaults to int24 (unlikely we'll have >16MM paths, but highly likely we'll have >65k paths)

line_num: identifier for which line num within a file 
	defaults to int24 (unlikely we'll have >16MM paths, but highly likely we'll have >65k docs per path)

	where signature needs to be big enough to not allow collisions

*/


use std::cmp::{PartialEq, Eq};
use std::hash::{Hash, Hasher};
use dashmap::DashMap;

/*=========================================================
=                         Helpful byte math               =
=========================================================*/


pub(crate) fn to_byte_size(n: usize) -> usize {
	// Calculates number of bytes needed to store n unique elements
	(f64::log2(n as f64) / 8.0 + 1.0).ceil() as usize
}

pub(crate) fn compute_sig_size(num_docs: usize) -> usize {
	// Computes how many bytes needed to store the signatures such that collisions don't occur
	// Overflow might be tricky here so need to do math
	// log(n * (n-1) /2) = log(n) + log(n-1) - 1
	let log_num_pairs = f64::log2(num_docs as f64) + f64::log2(num_docs as f64 - 1.0) - 1.0;
	(log_num_pairs / 8.0 + 1.0).ceil() as usize
}



/*=========================================================
=                        Nonstandard int types            =
=========================================================*/
pub(crate) trait IntValue: Hash + Eq {
    fn new(value: usize) -> Self where Self: Sized;
    fn from_bytes(value: Vec<u8>) -> Self where Self: Sized;
    fn as_bytes(&self) -> &[u8];
    fn as_usize(&self) -> usize;
}

#[derive(PartialEq, Eq)]
#[derive(Clone, Copy)]
#[derive(Debug)]
pub(crate) struct IntN<const N: usize> {
    bytes: [u8; N],
}

impl<const N: usize> IntValue for IntN<N> {
    fn new(value: usize) -> Self {
        let mut bytes = [0; N];
        for i in 0..N {
            bytes[i] = (value >> ((N - 1 - i) * 8)) as u8;
        }
        IntN { bytes }
    }

    fn from_bytes(value: Vec<u8>) -> IntN<N> {
        let mut bytes = [0; N];
        for i in 0..N {
            bytes[i] = value[i];
        }
        IntN { bytes }
    }

    fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }    

    fn as_usize(&self) -> usize {
	    let mut result: usize = 0;
	    for &byte in &self.bytes {
	        result = (result << 8) | byte as usize;
	    }
	    result
    }
}

impl<const N: usize> Hash for IntN<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bytes.hash(state);
    }
}


#[derive(Clone, Debug)]
pub(crate) enum IntValueEnum {
	 Int8(IntN<1>),
	 Int16(IntN<2>),
	 Int24(IntN<3>),
	 Int32(IntN<4>),
	 Int40(IntN<5>),
	 Int48(IntN<6>),
	 Int56(IntN<7>),
	 Int64(IntN<8>),
	 Int72(IntN<9>),
	 Int80(IntN<10>),
	 Int88(IntN<11>),
	 Int96(IntN<12>),
	 Int104(IntN<13>),
	 Int112(IntN<14>),
	 Int120(IntN<15>),
	 Int128(IntN<16>),
}

impl IntValueEnum {
    pub(crate) fn new(value: usize, num_bytes: usize) -> Self {
        match num_bytes {
			1 => IntValueEnum::Int8(IntN::<1>::new(value)),
			2 => IntValueEnum::Int16(IntN::<2>::new(value)),
			3 => IntValueEnum::Int24(IntN::<3>::new(value)),
			4 => IntValueEnum::Int32(IntN::<4>::new(value)),
			5 => IntValueEnum::Int40(IntN::<5>::new(value)),
			6 => IntValueEnum::Int48(IntN::<6>::new(value)),
			7 => IntValueEnum::Int56(IntN::<7>::new(value)),
			8 => IntValueEnum::Int64(IntN::<8>::new(value)),
			9 => IntValueEnum::Int72(IntN::<9>::new(value)),
			10 => IntValueEnum::Int80(IntN::<10>::new(value)),
			11 => IntValueEnum::Int88(IntN::<11>::new(value)),
			12 => IntValueEnum::Int96(IntN::<12>::new(value)),
			13 => IntValueEnum::Int104(IntN::<13>::new(value)),
			14 => IntValueEnum::Int112(IntN::<14>::new(value)),
			15 => IntValueEnum::Int120(IntN::<15>::new(value)),
			16 => IntValueEnum::Int128(IntN::<16>::new(value)),
            // Add more cases for other IntN types
            _ => panic!("Unsupported IntN type"),
        }
    }

    pub(crate) fn from_bytes(value: Vec<u8>, num_bytes: usize) -> Self {
        match num_bytes {
			1 => IntValueEnum::Int8(IntN::<1>::from_bytes(value)),
			2 => IntValueEnum::Int16(IntN::<2>::from_bytes(value)),
			3 => IntValueEnum::Int24(IntN::<3>::from_bytes(value)),
			4 => IntValueEnum::Int32(IntN::<4>::from_bytes(value)),
			5 => IntValueEnum::Int40(IntN::<5>::from_bytes(value)),
			6 => IntValueEnum::Int48(IntN::<6>::from_bytes(value)),
			7 => IntValueEnum::Int56(IntN::<7>::from_bytes(value)),
			8 => IntValueEnum::Int64(IntN::<8>::from_bytes(value)),
			9 => IntValueEnum::Int72(IntN::<9>::from_bytes(value)),
			10 => IntValueEnum::Int80(IntN::<10>::from_bytes(value)),
			11 => IntValueEnum::Int88(IntN::<11>::from_bytes(value)),
			12 => IntValueEnum::Int96(IntN::<12>::from_bytes(value)),
			13 => IntValueEnum::Int104(IntN::<13>::from_bytes(value)),
			14 => IntValueEnum::Int112(IntN::<14>::from_bytes(value)),
			15 => IntValueEnum::Int120(IntN::<15>::from_bytes(value)),
			16 => IntValueEnum::Int128(IntN::<16>::from_bytes(value)),
            // Add more cases for other IntN types
            _ => panic!("Unsupported IntN type"),
        }
    }

    pub(crate) fn as_usize(&self) -> usize {
        match self {
            IntValueEnum::Int8(value) => value.as_usize(),
            IntValueEnum::Int16(value) => value.as_usize(),
            IntValueEnum::Int24(value) => value.as_usize(),
            IntValueEnum::Int32(value) => value.as_usize(),
            IntValueEnum::Int40(value) => value.as_usize(),
            IntValueEnum::Int48(value) => value.as_usize(),
            IntValueEnum::Int56(value) => value.as_usize(),
            IntValueEnum::Int64(value) => value.as_usize(),
            IntValueEnum::Int72(value) => value.as_usize(),
            IntValueEnum::Int80(value) => value.as_usize(),
            IntValueEnum::Int88(value) => value.as_usize(),
            IntValueEnum::Int96(value) => value.as_usize(),
            IntValueEnum::Int104(value) => value.as_usize(),
            IntValueEnum::Int112(value) => value.as_usize(),
            IntValueEnum::Int120(value) => value.as_usize(),
            IntValueEnum::Int128(value) => value.as_usize(),
            // Add more cases for other IntN types
        }

    }


}

impl PartialEq for IntValueEnum {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
			(IntValueEnum::Int8(a), IntValueEnum::Int8(b)) => a == b,
			(IntValueEnum::Int16(a), IntValueEnum::Int16(b)) => a == b,
			(IntValueEnum::Int24(a), IntValueEnum::Int24(b)) => a == b,
			(IntValueEnum::Int32(a), IntValueEnum::Int32(b)) => a == b,
			(IntValueEnum::Int40(a), IntValueEnum::Int40(b)) => a == b,
			(IntValueEnum::Int48(a), IntValueEnum::Int48(b)) => a == b,
			(IntValueEnum::Int56(a), IntValueEnum::Int56(b)) => a == b,
			(IntValueEnum::Int64(a), IntValueEnum::Int64(b)) => a == b,
			(IntValueEnum::Int72(a), IntValueEnum::Int72(b)) => a == b,
			(IntValueEnum::Int80(a), IntValueEnum::Int80(b)) => a == b,
			(IntValueEnum::Int88(a), IntValueEnum::Int88(b)) => a == b,
			(IntValueEnum::Int96(a), IntValueEnum::Int96(b)) => a == b,
			(IntValueEnum::Int104(a), IntValueEnum::Int104(b)) => a == b,
			(IntValueEnum::Int112(a), IntValueEnum::Int112(b)) => a == b,
			(IntValueEnum::Int120(a), IntValueEnum::Int120(b)) => a == b,
			(IntValueEnum::Int128(a), IntValueEnum::Int128(b)) => a == b,
            // Add more cases for other IntN types
            _ => panic!("Unsupported IntN type"),
        }
    }
}

impl Eq for IntValueEnum {}

impl Hash for IntValueEnum {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            IntValueEnum::Int8(a) => a.hash(state),
            IntValueEnum::Int16(a) => a.hash(state),
            IntValueEnum::Int24(a) => a.hash(state),
            IntValueEnum::Int32(a) => a.hash(state),
            IntValueEnum::Int40(a) => a.hash(state),
            IntValueEnum::Int48(a) => a.hash(state),
            IntValueEnum::Int56(a) => a.hash(state),
            IntValueEnum::Int64(a) => a.hash(state),
            IntValueEnum::Int72(a) => a.hash(state),
            IntValueEnum::Int80(a) => a.hash(state),
            IntValueEnum::Int88(a) => a.hash(state),
            IntValueEnum::Int96(a) => a.hash(state),
            IntValueEnum::Int104(a) => a.hash(state),
            IntValueEnum::Int112(a) => a.hash(state),
            IntValueEnum::Int120(a) => a.hash(state),
            IntValueEnum::Int128(a) => a.hash(state),
            // Add more cases for other IntN types
        }
    }
}




/*==========================================================
=                     Band Storage Config.                 =
==========================================================*/

pub struct BandStorageConfig {
	pub sig_size: usize,
	pub path_size: usize,
	pub line_size: usize,
}

pub(crate) type BandStorage = DashMap<u64, DashMap<IntValueEnum, Vec<(IntValueEnum, IntValueEnum)>>>;




