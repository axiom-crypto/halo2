//! # halo2-axiom
//! This is a fork of <https://github.com/privacy-scaling-explorations/halo2>, which is itself a fork of ZCash's "halo2_proofs" crate.
//! This fork uses the KZG polynomial commitment scheme for the proving backend.
//! Publishing this crate for better versioning in Axiom's production usage.

#![cfg_attr(docsrs, feature(doc_cfg))]
// The actual lints we want to disable.
#![allow(clippy::op_ref, clippy::many_single_char_names)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
// #![deny(missing_docs)]
// #![deny(unsafe_code)]
#![feature(associated_type_defaults)]

#[cfg(feature = "counter")]
#[macro_use]
extern crate lazy_static;

#[cfg(feature = "counter")]
use lazy_static::lazy_static;

#[cfg(feature = "counter")]
use std::sync::Mutex;

#[cfg(feature = "counter")]
use std::collections::BTreeMap;

#[cfg(feature = "counter")]
lazy_static! {
    static ref FFT_COUNTER: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());
    static ref MSM_COUNTER: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());
}

pub mod arithmetic;
pub mod circuit;
pub use halo2curves;
pub mod fft;
mod multicore;
pub mod plonk;
pub mod poly;
pub mod transcript;

pub mod dev;
mod helpers;
pub use helpers::SerdeFormat;
