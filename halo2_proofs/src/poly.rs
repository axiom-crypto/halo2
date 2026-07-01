//! Contains utilities for performing arithmetic over univariate polynomials in
//! various forms, including computing commitments to them and provably opening
//! the committed polynomials at arbitrary points.

use std::fmt::Debug;
use std::io;
use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, Range, RangeFrom, RangeFull, Sub};

use crate::arithmetic::parallelize;
use crate::helpers::SerdePrimeField;
use crate::plonk::Assigned;
use crate::SerdeFormat;

#[cfg(feature = "multicore")]
use crate::multicore::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator, ParallelSlice,
};
use group::ff::{BatchInvert, Field};
use halo2curves::serde::SerdeObject;

/// Generic commitment scheme structures
pub mod commitment;
mod domain;
mod query;
mod strategy;

/// Inner product argument commitment scheme
pub mod ipa;

/// KZG commitment scheme
pub mod kzg;

#[cfg(test)]
mod multiopen_test;

pub use domain::*;
pub use query::{ProverQuery, VerifierQuery};
pub use strategy::{Guard, VerificationStrategy};

/// This is an error that could occur during proving or circuit synthesis.
// TODO: these errors need to be cleaned up
#[derive(Debug)]
pub enum Error {
    /// OpeningProof is not well-formed
    OpeningError,
    /// Caller needs to re-sample a point
    SamplingError,
}

/// The basis over which a polynomial is described.
pub trait Basis: Copy + Debug + Send + Sync {}

/// The polynomial is defined as coefficients
#[derive(Clone, Copy, Debug)]
pub struct Coeff;
impl Basis for Coeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials
#[derive(Clone, Copy, Debug)]
pub struct LagrangeCoeff;
impl Basis for LagrangeCoeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials in
/// an extended size domain which supports multiplication
#[derive(Clone, Copy, Debug)]
pub struct ExtendedLagrangeCoeff;
impl Basis for ExtendedLagrangeCoeff {}

/// Residency marker for a [`Polynomial`]. Implementors associate the backing
/// container type via the [`Storage::Backing`] GAT: `Vec<F>` for [`Host`].
/// Generic over `F` so a single marker can type every scalar choice the prover
/// instantiates.
pub trait Storage: 'static {
    /// Backing container holding the polynomial's coefficients.
    type Backing<F>;

    /// Length of the backing container in elements.
    fn backing_len<F>(b: &Self::Backing<F>) -> usize;

    /// Compile-time tag distinguishing the storage flavours. Used by the few
    /// code paths that are generic over `S` and want to take a runtime-fast
    /// branch without virtual dispatch (the optimiser folds the branch since
    /// `IS_DEVICE` is `const`).
    const IS_DEVICE: bool;
}

/// Marker indicating a host-resident polynomial whose coefficients live in a
/// `Vec<F>`.
#[derive(Clone, Copy, Debug)]
pub struct Host;

impl Storage for Host {
    type Backing<F> = Vec<F>;
    fn backing_len<F>(b: &Vec<F>) -> usize {
        b.len()
    }
    const IS_DEVICE: bool = false;
}

/// Represents a univariate polynomial defined over a field and a particular
/// basis, parameterised by its storage residency.
pub struct Polynomial<F, B, S: Storage = Host> {
    storage: S::Backing<F>,
    _marker: PhantomData<B>,
}

impl<F, B, S: Storage> Debug for Polynomial<F, B, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Polynomial")
            .field("len", &S::backing_len::<F>(&self.storage))
            .field("residency", &if S::IS_DEVICE { "Device" } else { "Host" })
            .finish()
    }
}

impl<F: Clone, B> Clone for Polynomial<F, B, Host> {
    fn clone(&self) -> Self {
        Self { storage: self.storage.clone(), _marker: PhantomData }
    }
}

impl<F, B, S: Storage> Polynomial<F, B, S> {
    /// Construct a polynomial directly from its backing container. This is the
    /// generic seam that lets out-of-crate storage backends build a
    /// `Polynomial` for any `S`.
    pub fn from_backing(backing: S::Backing<F>) -> Self {
        Self { storage: backing, _marker: PhantomData }
    }

    /// Borrow the backing container.
    pub fn backing(&self) -> &S::Backing<F> {
        &self.storage
    }

    /// Mutably borrow the backing container.
    pub fn backing_mut(&mut self) -> &mut S::Backing<F> {
        &mut self.storage
    }

    /// Consume the polynomial and return the owned backing container.
    pub fn into_backing(self) -> S::Backing<F> {
        self.storage
    }

    /// Number of coefficients.
    pub fn len(&self) -> usize {
        S::backing_len::<F>(&self.storage)
    }

    /// `true` if there are no coefficients.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets the size of this polynomial in terms of the number of
    /// coefficients used to describe it.
    pub fn num_coeffs(&self) -> usize {
        self.len()
    }
}

impl<F, B> Polynomial<F, B, Host> {
    /// Construct a host-resident polynomial directly from `Vec<F>`.
    pub fn new(values: Vec<F>) -> Self {
        Self { storage: values, _marker: PhantomData }
    }

    /// Direct host slice accessor.
    pub fn values(&self) -> &[F] {
        self.storage.as_slice()
    }

    /// Direct mutable host slice accessor.
    pub fn values_mut(&mut self) -> &mut [F] {
        self.storage.as_mut_slice()
    }

    /// Consume the polynomial and return the owned `Vec<F>` of host
    /// coefficients.
    pub fn into_values(self) -> Vec<F> {
        self.storage
    }

    /// Iterate over the values, which are either in coefficient or evaluation
    /// form depending on the basis `B`.
    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.storage.iter()
    }

    /// Iterate over the values mutably, which are either in coefficient or
    /// evaluation form depending on the basis `B`.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut F> {
        self.storage.iter_mut()
    }
}

impl<F, B> Index<usize> for Polynomial<F, B, Host> {
    type Output = F;

    fn index(&self, index: usize) -> &F {
        &self.values()[index]
    }
}

impl<F, B> IndexMut<usize> for Polynomial<F, B, Host> {
    fn index_mut(&mut self, index: usize) -> &mut F {
        &mut self.values_mut()[index]
    }
}

impl<F, B> Index<Range<usize>> for Polynomial<F, B, Host> {
    type Output = [F];

    fn index(&self, index: Range<usize>) -> &[F] {
        &self.values()[index]
    }
}

impl<F, B> Index<RangeFrom<usize>> for Polynomial<F, B, Host> {
    type Output = [F];

    fn index(&self, index: RangeFrom<usize>) -> &[F] {
        &self.values()[index]
    }
}

impl<F, B> IndexMut<Range<usize>> for Polynomial<F, B, Host> {
    fn index_mut(&mut self, index: Range<usize>) -> &mut [F] {
        &mut self.values_mut()[index]
    }
}

impl<F, B> IndexMut<RangeFrom<usize>> for Polynomial<F, B, Host> {
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [F] {
        &mut self.values_mut()[index]
    }
}

impl<F, B> Index<RangeFull> for Polynomial<F, B, Host> {
    type Output = [F];

    fn index(&self, _index: RangeFull) -> &[F] {
        self.values()
    }
}

impl<F, B> IndexMut<RangeFull> for Polynomial<F, B, Host> {
    fn index_mut(&mut self, index: RangeFull) -> &mut [F] {
        &mut self.values_mut()[index]
    }
}

impl<F, B> Deref for Polynomial<F, B, Host> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        self.values()
    }
}

impl<F, B> DerefMut for Polynomial<F, B, Host> {
    fn deref_mut(&mut self) -> &mut [F] {
        self.values_mut()
    }
}

impl<F: SerdePrimeField, B> Polynomial<F, B> {
    /// Reads polynomial from buffer using `SerdePrimeField::read`.
    pub(crate) fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> Self {
        let mut poly_len = [0u8; 4];
        reader.read_exact(&mut poly_len).unwrap();
        let poly_len = u32::from_be_bytes(poly_len) as usize;

        let values = match format {
            // Raw formats store `F`'s in-memory representation verbatim, so read
            // the whole polynomial into the destination `Vec<F>` with a single
            // `read_exact` rather than ~4 per-element `Read` calls.
            SerdeFormat::RawBytes | SerdeFormat::RawBytesUnchecked => {
                let elem_size = std::mem::size_of::<F>();
                // The direct read is only sound if the serialized element width
                // matches `F`'s in-memory size.
                assert_eq!(
                    elem_size,
                    F::ZERO.to_raw_bytes().len(),
                    "raw element size does not match in-memory size of F"
                );

                // Read into uninitialized spare capacity to avoid the zero-fill
                // of `vec![F::ZERO; n]`.
                let mut values: Vec<F> = Vec::with_capacity(poly_len);
                // SAFETY: `with_capacity` reserved `poly_len * elem_size` bytes;
                // a `*mut u8` view is always well-aligned and `read_exact` fully
                // initializes the region. On little-endian targets those bytes
                // are then valid `F` -- the layout/endianness assumption
                // `read_raw`/`from_raw_bytes` already rely on.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(
                        values.as_mut_ptr() as *mut u8,
                        poly_len * elem_size,
                    )
                };
                reader.read_exact(dst).unwrap();

                // `RawBytes` additionally requires each element to be < modulus;
                // check in parallel since the bytes are already in memory.
                if matches!(format, SerdeFormat::RawBytes) {
                    #[cfg(feature = "multicore")]
                    let all_valid = dst
                        .par_chunks_exact(elem_size)
                        .all(|chunk| <F as SerdeObject>::from_raw_bytes(chunk).is_some());
                    #[cfg(not(feature = "multicore"))]
                    let all_valid = dst
                        .chunks_exact(elem_size)
                        .all(|chunk| <F as SerdeObject>::from_raw_bytes(chunk).is_some());
                    assert!(all_valid, "invalid field element: not less than modulus");
                }
                // SAFETY: `read_exact` filled all `poly_len` elements above.
                unsafe { values.set_len(poly_len) };
                values
            }
            SerdeFormat::Processed => {
                (0..poly_len).map(|_| F::read(reader, format).unwrap()).collect()
            }
        };

        Self::new(values)
    }

    /// Writes polynomial to buffer using `SerdePrimeField::write`.
    pub(crate) fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) {
        let values = self.values();
        writer.write_all(&(values.len() as u32).to_be_bytes()).unwrap();
        for value in values.iter() {
            value.write(writer, format).unwrap();
        }
    }
}

/// Invert each polynomial in place for memory efficiency
pub(crate) fn batch_invert_assigned<F: Field, PA>(
    assigned: Vec<PA>,
) -> Vec<Polynomial<F, LagrangeCoeff>>
where
    PA: Deref<Target = [Assigned<F>]> + Sync,
{
    if assigned.is_empty() {
        return vec![];
    }
    let n = assigned[0].as_ref().len();
    // 1d vector better for memory allocation
    let mut assigned_denominators: Vec<_> =
        assigned.iter().flat_map(|f| f.as_ref().iter().map(|value| value.denominator())).collect();

    assigned_denominators
        .iter_mut()
        // If the denominator is trivial, we can skip it, reducing the
        // size of the batch inversion.
        .filter_map(|d| d.as_mut())
        .batch_invert();

    #[cfg(feature = "multicore")]
    return assigned
        .par_iter()
        .zip(assigned_denominators.par_chunks(n))
        .map(|(poly, inv_denoms)| {
            debug_assert_eq!(inv_denoms.len(), poly.as_ref().len());
            Polynomial::new(
                poly.as_ref()
                    .iter()
                    .zip(inv_denoms.iter())
                    .map(|(a, inv_den)| a.numerator() * inv_den.unwrap_or(F::ONE))
                    .collect(),
            )
        })
        .collect();

    #[cfg(not(feature = "multicore"))]
    return assigned
        .iter()
        .zip(assigned_denominators.chunks(n))
        .map(|(poly, inv_denoms)| {
            debug_assert_eq!(inv_denoms.len(), poly.as_ref().len());
            Polynomial::new(
                poly.as_ref()
                    .iter()
                    .zip(inv_denoms.iter())
                    .map(|(a, inv_den)| a.numerator() * inv_den.unwrap_or(F::ONE))
                    .collect(),
            )
        })
        .collect();
}

impl<F: Field> Polynomial<Assigned<F>, LagrangeCoeff> {
    pub fn invert(
        &self,
        inv_denoms: impl Iterator<Item = F> + ExactSizeIterator,
    ) -> Polynomial<F, LagrangeCoeff> {
        let src = self.values();
        assert_eq!(inv_denoms.len(), src.len());
        let values: Vec<F> =
            src.iter().zip(inv_denoms).map(|(a, inv_den)| a.numerator() * inv_den).collect();
        Polynomial::new(values)
    }
}

impl<'a, F: Field, B: Basis> Add<&'a Polynomial<F, B>> for Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn add(mut self, rhs: &'a Polynomial<F, B>) -> Polynomial<F, B> {
        let rhs_slice = rhs.values();
        parallelize(self.values_mut(), |lhs, start| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs_slice[start..].iter()) {
                *lhs += *rhs;
            }
        });

        self
    }
}

impl<'a, F: Field, B: Basis> Sub<&'a Polynomial<F, B>> for Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn sub(mut self, rhs: &'a Polynomial<F, B>) -> Polynomial<F, B> {
        let rhs_slice = rhs.values();
        parallelize(self.values_mut(), |lhs, start| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs_slice[start..].iter()) {
                *lhs -= *rhs;
            }
        });

        self
    }
}

impl<F: Field> Polynomial<F, LagrangeCoeff> {
    /// Rotates the values in a Lagrange basis polynomial by `Rotation`
    pub fn rotate(&self, rotation: Rotation) -> Polynomial<F, LagrangeCoeff> {
        let mut values = self.values().to_vec();
        if rotation.0 < 0 {
            values.rotate_right((-rotation.0) as usize);
        } else {
            values.rotate_left(rotation.0 as usize);
        }
        Polynomial::new(values)
    }
}

impl<F: Field, B: Basis> Mul<F> for Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn mul(mut self, rhs: F) -> Polynomial<F, B> {
        if rhs == F::ZERO {
            return Polynomial::new(vec![F::ZERO; self.len()]);
        }
        if rhs == F::ONE {
            return self;
        }

        parallelize(self.values_mut(), |lhs, _| {
            for lhs in lhs.iter_mut() {
                *lhs *= rhs;
            }
        });

        self
    }
}

impl<'a, F: Field, B: Basis> Sub<F> for &'a Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn sub(self, rhs: F) -> Polynomial<F, B> {
        let mut res = self.clone();
        res.values_mut()[0] -= rhs;
        res
    }
}

/// Describes the relative rotation of a vector. Negative numbers represent
/// reverse (leftmost) rotations and positive numbers represent forward (rightmost)
/// rotations. Zero represents no rotation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Rotation(pub i32);

impl Rotation {
    /// The current location in the evaluation domain
    pub fn cur() -> Rotation {
        Rotation(0)
    }

    /// The previous location in the evaluation domain
    pub fn prev() -> Rotation {
        Rotation(-1)
    }

    /// The next location in the evaluation domain
    pub fn next() -> Rotation {
        Rotation(1)
    }
}
