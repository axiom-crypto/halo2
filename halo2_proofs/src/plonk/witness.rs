#[cfg(feature = "profile")]
use ark_std::{end_timer, start_timer};
use ff::{Field, WithSmallOrderMulGroup};
use group::Curve;
use rand_core::RngCore;

use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::RangeTo;

#[cfg(feature = "multicore")]
use crate::multicore::IndexedParallelIterator;
use crate::multicore::{IntoParallelIterator, ParallelIterator};

use super::{
    circuit::{
        sealed::{self},
        Advice, Any, Assignment, Challenge, Circuit, Column, ConstraintSystem, Fixed, FloorPlanner,
        Instance, Selector,
    },
    Error, ProvingKey,
};
use crate::{
    arithmetic::CurveAffine,
    circuit::Value,
    plonk::Assigned,
    poly::{
        commitment::{Blind, CommitmentScheme, Params, Prover},
        Basis, Coeff, LagrangeCoeff, Polynomial,
    },
};
use crate::{
    poly::batch_invert_assigned,
    transcript::{EncodedChallenge, TranscriptWrite},
};

/// Per-circuit instance state produced by phase-1 synthesis.
///
/// Mirrors the halo2-gpu fork's `InstanceSingle` shape so downstream crates can
/// name the same struct across cpu/cuda builds. `instance_polys` is the
/// coefficient-basis form of `instance_values` (populated eagerly during phase 1).
#[derive(Debug)]
pub struct InstanceSingle<C: CurveAffine> {
    pub instance_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    pub instance_polys: Vec<Polynomial<C::Scalar, Coeff>>,
}

/// Per-circuit advice state produced by [`synthesize_witness`].
///
/// Mirrors the halo2-gpu fork's `AdviceSingle` shape (both bases, no blinds).
/// `advice_values` is the Lagrange-basis witness; `advice_polys` is the same
/// data in coefficient basis. Blinds are drawn but not retained here — they
/// are only needed on the internal proving path.
#[derive(Debug)]
pub struct AdviceSingle<C: CurveAffine> {
    pub advice_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    pub advice_polys: Vec<Polynomial<C::Scalar, Coeff>>,
}

#[derive(Clone)]
pub(super) struct AdviceCommitted<C: CurveAffine, B: Basis> {
    pub advice_polys: Vec<Polynomial<C::Scalar, B>>,
    pub advice_blinds: Vec<Blind<C::Scalar>>,
}

/// The shape of advice columns accepted by [`super::create_proof_from_advice`]:
/// one `Vec<F>` per physical advice column, each of length `params.n()`.
pub type AdviceColumns<F> = Vec<Vec<F>>;

/// Materialize one [`InstanceSingle`] from a slice of instance-column value
/// vectors, zero-padding to `n = params.n()`. Panics if a column is longer
/// than the usable region (matches the check in [`super::create_proof`]).
pub(super) fn build_instance_single<Scheme: CommitmentScheme>(
    params: &Scheme::ParamsProver,
    domain: &crate::poly::EvaluationDomain<Scheme::Scalar>,
    meta: &ConstraintSystem<Scheme::Scalar>,
    instance: &[&[Scheme::Scalar]],
) -> InstanceSingle<Scheme::Curve>
where
    Scheme::Scalar: WithSmallOrderMulGroup<3>,
{
    let n = params.n() as usize;
    let instance_values = instance
        .iter()
        .map(|values| {
            let mut poly = domain.empty_lagrange();
            assert_eq!(poly.len(), n);
            if values.len() > (poly.len() - (meta.blinding_factors() + 1)) {
                panic!("Error::InstanceTooLarge");
            }
            for (poly, value) in poly.iter_mut().zip(values.iter()) {
                *poly = *value;
            }
            poly
        })
        .collect::<Vec<_>>();

    let instance_polys: Vec<_> = instance_values
        .iter()
        .map(|poly| {
            let lagrange_vec = domain.lagrange_from_vec(poly.to_vec());
            domain.lagrange_to_coeff(lagrange_vec)
        })
        .collect();

    InstanceSingle {
        instance_values,
        instance_polys,
    }
}

/// Bucket advice columns and challenges by their phase index (0/1/2).
pub(super) fn column_and_challenge_indices<F: Field>(
    meta: &ConstraintSystem<F>,
) -> ([Vec<usize>; 3], [Vec<usize>; 3]) {
    let mut column_indices = [(); 3].map(|_| vec![]);
    for (index, phase) in meta.advice_column_phase.iter().enumerate() {
        column_indices[phase.to_u8() as usize].push(index);
    }
    let mut challenge_indices = [(); 3].map(|_| vec![]);
    for (index, phase) in meta.challenge_phase.iter().enumerate() {
        challenge_indices[phase.to_u8() as usize].push(index);
    }
    (column_indices, challenge_indices)
}

/// Drives `Circuit::synthesize` for each circuit, running the same
/// commit / challenge-squeeze sequence [`super::create_proof`] performs in
/// phase 1. Returns per-circuit committed advice (Lagrange basis, with blinds)
/// and the flat per-index challenge vector.
///
/// Shared between [`super::create_proof`] and [`synthesize_witness`].
pub(super) fn run_phase1_synthesis<
    'params,
    'a,
    Scheme: CommitmentScheme,
    P: Prover<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    R: RngCore + 'a,
    T: TranscriptWrite<Scheme::Curve, E>,
    ConcreteCircuit: Circuit<Scheme::Scalar>,
>(
    params: &'params Scheme::ParamsProver,
    pk: &ProvingKey<Scheme::Curve>,
    circuits: &[ConcreteCircuit],
    instances: &[&'a [&'a [Scheme::Scalar]]],
    instance: &[InstanceSingle<Scheme::Curve>],
    rng: &mut R,
    mut transcript: &'a mut T,
) -> Result<
    (
        Vec<AdviceCommitted<Scheme::Curve, LagrangeCoeff>>,
        Vec<Scheme::Scalar>,
    ),
    Error,
>
where
    Scheme::Scalar: Hash + WithSmallOrderMulGroup<3>,
    <Scheme as CommitmentScheme>::ParamsProver: Sync,
{
    let domain = &pk.vk.domain;
    let meta = &pk.vk.cs;

    let mut meta_for_config = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut meta_for_config, circuits[0].params());
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut meta_for_config);

    let (column_indices, challenge_indices) = column_and_challenge_indices(meta);

    let mut advice = Vec::with_capacity(instances.len());
    let mut challenges = HashMap::<usize, Scheme::Scalar>::with_capacity(meta.num_challenges);

    let unusable_rows_start = params.n() as usize - (meta.blinding_factors() + 1);
    let phases = meta.phases().collect::<Vec<_>>();
    let num_phases = phases.len();
    // WARNING: this will currently not work if `circuits` has more than 1 circuit
    // because the original API squeezes the challenges for a phase after running all circuits
    // once in that phase.
    if num_phases > 1 {
        assert_eq!(
            circuits.len(),
            1,
            "New challenge API doesn't work with multiple circuits yet"
        );
    }
    for ((circuit, circuit_instances), instance_single) in
        circuits.iter().zip(instances).zip(instance.iter())
    {
        let mut witness: WitnessCollection<Scheme, P, _, E, _, _> = WitnessCollection {
            params,
            current_phase: phases[0],
            advice: vec![domain.empty_lagrange_assigned(); meta.num_advice_columns],
            instances: circuit_instances,
            challenges: &mut challenges,
            usable_rows: ..unusable_rows_start,
            advice_single: AdviceCommitted::<Scheme::Curve, LagrangeCoeff> {
                advice_polys: vec![domain.empty_lagrange(); meta.num_advice_columns],
                advice_blinds: vec![Blind::default(); meta.num_advice_columns],
            },
            instance_single,
            rng,
            transcript: &mut transcript,
            column_indices: column_indices.clone(),
            challenge_indices: challenge_indices.clone(),
            unusable_rows_start,
            _marker: PhantomData,
        };

        // while loop is for compatibility with circuits that do not use the new `next_phase` API to manage phases
        // If the circuit uses the new API, then the while loop will only execute once
        while witness.current_phase.to_u8() < num_phases as u8 {
            #[cfg(feature = "profile")]
            let syn_time = start_timer!(|| format!(
                "Synthesize time starting from phase {} (synthesize may cross multiple phases)",
                witness.current_phase.to_u8()
            ));
            // Synthesize the circuit to obtain the witness and other information.
            ConcreteCircuit::FloorPlanner::synthesize(
                &mut witness,
                circuit,
                config.clone(),
                meta.constants.clone(),
            )
            .unwrap();
            #[cfg(feature = "profile")]
            end_timer!(syn_time);
            if witness.current_phase.to_u8() < num_phases as u8 {
                witness.next_phase();
            }
        }
        advice.push(witness.advice_single);
    }

    assert_eq!(challenges.len(), meta.num_challenges);
    let challenges = (0..meta.num_challenges)
        .map(|index| challenges.remove(&index).unwrap())
        .collect::<Vec<_>>();

    Ok((advice, challenges))
}

pub(super) struct WitnessCollection<'params, 'a, 'b, Scheme, P, C, E, R, T>
where
    Scheme: CommitmentScheme<Curve = C>,
    P: Prover<'params, Scheme>,
    C: CurveAffine,
    E: EncodedChallenge<C>,
    R: RngCore + 'a,
    T: TranscriptWrite<C, E>,
{
    pub params: &'params Scheme::ParamsProver,
    pub current_phase: sealed::Phase,
    pub advice: Vec<Polynomial<Assigned<C::Scalar>, LagrangeCoeff>>,
    pub challenges: &'b mut HashMap<usize, C::Scalar>,
    pub instances: &'b [&'a [C::Scalar]],
    pub usable_rows: RangeTo<usize>,
    pub advice_single: AdviceCommitted<C, LagrangeCoeff>,
    pub instance_single: &'b InstanceSingle<C>,
    pub rng: &'b mut R,
    pub transcript: &'b mut &'a mut T,
    pub column_indices: [Vec<usize>; 3],
    pub challenge_indices: [Vec<usize>; 3],
    pub unusable_rows_start: usize,
    pub _marker: PhantomData<(P, E)>,
}

impl<'params, 'a, 'b, F, Scheme, P, C, E, R, T> Assignment<F>
    for WitnessCollection<'params, 'a, 'b, Scheme, P, C, E, R, T>
where
    F: Field,
    Scheme: CommitmentScheme<Curve = C>,
    P: Prover<'params, Scheme>,
    C: CurveAffine<ScalarExt = F>,
    E: EncodedChallenge<C>,
    R: RngCore,
    T: TranscriptWrite<C, E>,
    <Scheme as CommitmentScheme>::ParamsProver: Sync,
{
    fn enter_region<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about regions in this context.
    }

    fn exit_region(&mut self) {
        // Do nothing; we don't care about regions in this context.
    }

    fn enable_selector<A, AR>(&mut self, _: A, _: &Selector, _: usize) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // We only care about advice columns here

        Ok(())
    }

    fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // Do nothing
    }

    fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.params.k()));
        }

        self.instances
            .get(column.index())
            .and_then(|column| column.get(row))
            .map(|v| Value::known(*v))
            .ok_or(Error::BoundsFailure)
    }

    fn assign_advice<'v>(
        //<V, VR, A, AR>(
        &mut self,
        //_: A,
        column: Column<Advice>,
        row: usize,
        to: Value<Assigned<F>>,
    ) -> Value<&'v Assigned<F>> {
        // debug_assert_eq!(self.current_phase, column.column_type().phase);

        debug_assert!(
            self.usable_rows.contains(&row),
            "{:?}",
            Error::not_enough_rows_available(self.params.k())
        );

        let advice_get_mut = self
            .advice
            .get_mut(column.index())
            .expect("Not enough advice columns")
            .get_mut(row)
            .expect("Not enough rows");
        // We can get another 3-4% decrease in witness gen time by using the following unsafe code, but this skips all array bound checks so we should use it only if the performance gain is really necessary:
        /*
        let advice_get_mut = unsafe {
            self.advice
                .get_unchecked_mut(column.index())
                .get_unchecked_mut(row)
        };
        */
        *advice_get_mut = to
            .assign()
            .expect("No Value::unknown() in advice column allowed during create_proof");
        let immutable_raw_ptr = advice_get_mut as *const Assigned<F>;
        Value::known(unsafe { &*immutable_raw_ptr })
    }

    fn assign_fixed(&mut self, _: Column<Fixed>, _: usize, _: Assigned<F>) {
        // We only care about advice columns here
    }

    fn copy(&mut self, _: Column<Any>, _: usize, _: Column<Any>, _: usize) {
        // We only care about advice columns here
    }

    fn fill_from_row(
        &mut self,
        _: Column<Fixed>,
        _: usize,
        _: Value<Assigned<F>>,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn get_challenge(&self, challenge: Challenge) -> Value<F> {
        self.challenges
            .get(&challenge.index())
            .cloned()
            .map(Value::known)
            .unwrap_or_else(Value::unknown)
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self, _: Option<String>) {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn next_phase(&mut self) {
        let phase = self.current_phase.to_u8() as usize;
        #[cfg(feature = "profile")]
        let start1 = start_timer!(|| format!("Phase {phase} inversion and MSM commitment"));
        if phase == 0 {
            // Absorb instances into transcript.
            // Do this here and not earlier in case we want to be able to mutate
            // the instances during synthesize in FirstPhase in the future
            if !P::QUERY_INSTANCE {
                for values in self.instances.iter() {
                    for value in values.iter() {
                        self.transcript
                            .common_scalar(*value)
                            .expect("Absorbing instance value to transcript failed");
                    }
                }
            } else {
                let instance_commitments_projective: Vec<_> =
                    (&self.instance_single.instance_values)
                        .into_par_iter()
                        .map(|poly| self.params.commit_lagrange(poly, Blind::default()))
                        .collect();
                let mut instance_commitments =
                    vec![C::identity(); instance_commitments_projective.len()];
                C::CurveExt::batch_normalize(
                    &instance_commitments_projective,
                    &mut instance_commitments,
                );
                let instance_commitments = instance_commitments;
                drop(instance_commitments_projective);

                for commitment in &instance_commitments {
                    self.transcript
                        .common_point(*commitment)
                        .expect("Absorbing instance commitment to transcript failed");
                }
            }
        }
        // Commit the advice columns in the current phase
        let mut advice_values = batch_invert_assigned(
            self.column_indices
                .get(phase)
                .expect("The API only supports 3 phases right now")
                .iter()
                .map(|column_index| &self.advice[*column_index][..])
                .collect(),
        );
        // Add blinding factors to advice columns
        for advice_values in &mut advice_values {
            for cell in &mut advice_values[self.unusable_rows_start..] {
                *cell = F::random(&mut self.rng);
            }
        }
        // Compute commitments to advice column polynomials
        let blinds: Vec<_> = advice_values
            .iter()
            .map(|_| Blind(F::random(&mut self.rng)))
            .collect();
        let advice_commitments_projective: Vec<_> = (&advice_values)
            .into_par_iter()
            .zip((&blinds).into_par_iter())
            .map(|(poly, blind)| self.params.commit_lagrange(poly, *blind))
            .collect();
        let mut advice_commitments = vec![C::identity(); advice_commitments_projective.len()];
        C::CurveExt::batch_normalize(&advice_commitments_projective, &mut advice_commitments);
        let advice_commitments = advice_commitments;
        drop(advice_commitments_projective);

        for commitment in &advice_commitments {
            self.transcript
                .write_point(*commitment)
                .expect("Absorbing advice commitment to transcript failed");
        }
        for ((column_index, advice_poly), blind) in self.column_indices[phase]
            .iter()
            .zip(advice_values)
            .zip(blinds)
        {
            self.advice_single.advice_polys[*column_index] = advice_poly;
            self.advice_single.advice_blinds[*column_index] = blind;
        }
        for challenge_index in self.challenge_indices[phase].iter() {
            let existing = self.challenges.insert(
                *challenge_index,
                *self.transcript.squeeze_challenge_scalar::<()>(),
            );
            assert!(existing.is_none());
        }
        self.current_phase = self.current_phase.next();
        #[cfg(feature = "profile")]
        end_timer!(start1);
    }
}

/// Runs the phase-1 half of proof generation — the same synthesize / commit /
/// challenge-squeeze steps [`super::create_proof`] performs internally — and
/// returns the resulting instance/advice singles and per-phase challenges
/// without continuing to lookups, permutations, or the multiopen argument.
///
/// The transcript is left in the same state [`super::create_proof`] would
/// produce at the boundary (vk hashed, instances absorbed, per-phase advice
/// commitments written, per-phase challenges squeezed). Intended as a
/// diagnostic entry point — comparing an out-of-band witness generator's
/// advice against `Circuit::synthesize` — not for producing verifiable proofs.
///
/// Mirrors the halo2-gpu fork's `synthesize_witness` shape so downstream crates
/// can gate on the fork via a single import.
pub fn synthesize_witness<
    'params,
    'a,
    Scheme: CommitmentScheme,
    P: Prover<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    R: RngCore + 'a,
    T: TranscriptWrite<Scheme::Curve, E>,
    ConcreteCircuit: Circuit<Scheme::Scalar>,
>(
    params: &'params Scheme::ParamsProver,
    pk: &ProvingKey<Scheme::Curve>,
    circuits: &[ConcreteCircuit],
    instances: &[&'a [&'a [Scheme::Scalar]]],
    mut rng: R,
    transcript: &'a mut T,
) -> Result<
    (
        Vec<InstanceSingle<Scheme::Curve>>,
        Vec<AdviceSingle<Scheme::Curve>>,
        Vec<Scheme::Scalar>,
    ),
    Error,
>
where
    Scheme::Scalar: Hash + WithSmallOrderMulGroup<3>,
    <Scheme as CommitmentScheme>::ParamsProver: Sync,
{
    if circuits.len() != instances.len() {
        return Err(Error::InvalidInstances);
    }

    for instance in instances.iter() {
        if instance.len() != pk.vk.cs.num_instance_columns {
            return Err(Error::InvalidInstances);
        }
    }

    // Hash verification key into transcript
    pk.vk.hash_into(transcript)?;

    let domain = &pk.vk.domain;
    let meta = &pk.vk.cs;

    let instance: Vec<InstanceSingle<Scheme::Curve>> = instances
        .iter()
        .map(|instance| build_instance_single::<Scheme>(params, domain, meta, instance))
        .collect();

    let (advice, challenges) = run_phase1_synthesis::<Scheme, P, E, R, T, ConcreteCircuit>(
        params, pk, circuits, instances, &instance, &mut rng, transcript,
    )?;

    // Convert internal (Lagrange advice + blinds) into the public shape by
    // dropping blinds and computing the coefficient-basis advice via iFFT.
    let advice: Vec<AdviceSingle<Scheme::Curve>> = advice
        .into_iter()
        .map(|committed| {
            let AdviceCommitted {
                advice_polys: advice_values,
                advice_blinds: _,
            } = committed;
            let advice_polys = advice_values
                .iter()
                .map(|poly| {
                    let lagrange_vec = domain.lagrange_from_vec(poly.to_vec());
                    domain.lagrange_to_coeff(lagrange_vec)
                })
                .collect::<Vec<_>>();
            AdviceSingle {
                advice_values,
                advice_polys,
            }
        })
        .collect();

    Ok((instance, advice, challenges))
}
