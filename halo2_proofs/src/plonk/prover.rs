#[cfg(feature = "profile")]
use ark_std::{end_timer, start_timer};
use ff::{Field, WithSmallOrderMulGroup};
use group::{prime::PrimeCurveAffine, Curve};
use rand_core::RngCore;

use std::hash::Hash;
use std::iter;

#[cfg(feature = "multicore")]
use crate::multicore::IndexedParallelIterator;
use crate::multicore::{IntoParallelIterator, ParallelIterator};
use std::collections::HashMap;

use super::witness::{build_instance_single, AdviceCommitted, InstanceSingle};
use super::{
    circuit::Circuit, lookup, permutation, vanishing, witness, ChallengeBeta, ChallengeGamma,
    ChallengeTheta, ChallengeX, ChallengeY, Error, ProvingKey,
};

use crate::transcript::{EncodedChallenge, TranscriptWrite};
use crate::{
    arithmetic::{eval_polynomial, CurveAffine},
    poly::{
        commitment::{Blind, CommitmentScheme, Params, Prover},
        Coeff, LagrangeCoeff, Polynomial, ProverQuery,
    },
};

/// This creates a proof for the provided `circuit` when given the public
/// parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The provided `instances`
/// are zero-padded internally.
pub fn create_proof<
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
    instances: &[&[&'a [Scheme::Scalar]]],
    mut rng: R,
    transcript: &'a mut T,
) -> Result<(), Error>
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

    #[cfg(feature = "profile")]
    let phase1_time = start_timer!(|| "Phase 1: Witness assignment and MSM commitments");
    let (advice, challenges) = witness::run_phase1_synthesis::<Scheme, P, E, R, T, ConcreteCircuit>(
        params, pk, circuits, instances, &instance, &mut rng, transcript,
    )?;
    #[cfg(feature = "profile")]
    end_timer!(phase1_time);

    create_proof_from_committed::<Scheme, P, E, R, T>(
        params, pk, instance, advice, challenges, rng, transcript,
    )
}

/// Creates a proof from pre-computed advice columns and per-circuit instances.
///
/// Skips `Circuit::synthesize`: caller supplies advice values (one `Vec<Scalar>` of
/// length `params.n()` per column). Blinding-factor rows may be left zero — this
/// function overwrites them from `rng` in the same order [`create_proof`] would.
/// Restricted to single-circuit, single-phase circuits: multi-phase proving
/// interleaves challenge squeezes with synthesis and cannot be expressed as a
/// pre-synthesized advice hand-off.
pub fn create_proof_from_advice<
    'params,
    'a,
    Scheme: CommitmentScheme,
    P: Prover<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    R: RngCore + 'a,
    T: TranscriptWrite<Scheme::Curve, E>,
>(
    params: &'params Scheme::ParamsProver,
    pk: &ProvingKey<Scheme::Curve>,
    instances: &'a [&'a [Scheme::Scalar]],
    mut advice: witness::AdviceColumns<Scheme::Scalar>,
    mut rng: R,
    transcript: &'a mut T,
) -> Result<(), Error>
where
    Scheme::Scalar: Hash + WithSmallOrderMulGroup<3>,
    <Scheme as CommitmentScheme>::ParamsProver: Sync,
{
    let meta = &pk.vk.cs;
    if instances.len() != meta.num_instance_columns {
        return Err(Error::InvalidInstances);
    }
    let phases: Vec<_> = meta.phases().collect();
    assert_eq!(
        phases.len(),
        1,
        "create_proof_from_advice supports single-phase circuits only: multi-phase \
         challenge squeezes interleave with synthesis and cannot go through \
         create_proof_from_advice"
    );
    assert_eq!(
        advice.len(),
        meta.num_advice_columns,
        "create_proof_from_advice: advice column count mismatch"
    );
    let n = params.n() as usize;
    for col in advice.iter() {
        assert_eq!(
            col.len(),
            n,
            "create_proof_from_advice: advice column length must equal params.n()"
        );
    }

    // Hash verification key into transcript
    pk.vk.hash_into(transcript)?;

    let domain = &pk.vk.domain;

    // Materialize the single instance
    let instance_single = build_instance_single::<Scheme>(params, domain, meta, instances);

    // Absorb instances into transcript, mirroring `WitnessCollection::next_phase`
    // for phase 0.
    if !P::QUERY_INSTANCE {
        for values in instances.iter() {
            for value in values.iter() {
                transcript
                    .common_scalar(*value)
                    .expect("Absorbing instance value to transcript failed");
            }
        }
    } else {
        let instance_commitments_projective: Vec<_> = (&instance_single.instance_values)
            .into_par_iter()
            .map(|poly| params.commit_lagrange(poly, Blind::default()))
            .collect();
        let mut instance_commitments =
            vec![Scheme::Curve::identity(); instance_commitments_projective.len()];
        <Scheme::Curve as CurveAffine>::CurveExt::batch_normalize(
            &instance_commitments_projective,
            &mut instance_commitments,
        );
        drop(instance_commitments_projective);

        for commitment in &instance_commitments {
            transcript
                .common_point(*commitment)
                .expect("Absorbing instance commitment to transcript failed");
        }
    }

    // Fill blinding rows and draw commitment blinds in the same rng order as
    // `create_proof` (per-column tail blinders, then one Blind per column).
    let unusable_rows_start = n - (meta.blinding_factors() + 1);
    for col in advice.iter_mut() {
        for cell in &mut col[unusable_rows_start..] {
            *cell = Scheme::Scalar::random(&mut rng);
        }
    }
    let advice_blinds: Vec<Blind<Scheme::Scalar>> = advice
        .iter()
        .map(|_| Blind(Scheme::Scalar::random(&mut rng)))
        .collect();

    // Wrap advice as Lagrange polynomials.
    let advice_values: Vec<Polynomial<Scheme::Scalar, LagrangeCoeff>> = advice
        .into_iter()
        .map(|col| domain.lagrange_from_vec(col))
        .collect();

    // Commit to advice columns.
    let advice_commitments_projective: Vec<_> = (&advice_values)
        .into_par_iter()
        .zip((&advice_blinds).into_par_iter())
        .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
        .collect();
    let mut advice_commitments =
        vec![Scheme::Curve::identity(); advice_commitments_projective.len()];
    <Scheme::Curve as CurveAffine>::CurveExt::batch_normalize(
        &advice_commitments_projective,
        &mut advice_commitments,
    );
    drop(advice_commitments_projective);

    for commitment in &advice_commitments {
        transcript
            .write_point(*commitment)
            .expect("Absorbing advice commitment to transcript failed");
    }

    // Squeeze phase-0 challenges in constraint-system order.
    let (_column_indices, challenge_indices) = witness::column_and_challenge_indices(meta);
    let mut challenges_map = HashMap::<usize, Scheme::Scalar>::with_capacity(meta.num_challenges);
    for challenge_index in challenge_indices[0].iter() {
        let existing = challenges_map.insert(
            *challenge_index,
            *transcript.squeeze_challenge_scalar::<()>(),
        );
        assert!(existing.is_none());
    }
    assert_eq!(challenges_map.len(), meta.num_challenges);
    let challenges: Vec<Scheme::Scalar> = (0..meta.num_challenges)
        .map(|i| challenges_map.remove(&i).unwrap())
        .collect();

    let advice_single = AdviceCommitted::<Scheme::Curve, LagrangeCoeff> {
        advice_polys: advice_values,
        advice_blinds,
    };

    create_proof_from_committed::<Scheme, P, E, R, T>(
        params,
        pk,
        vec![instance_single],
        vec![advice_single],
        challenges,
        rng,
        transcript,
    )
}

/// Post-phase-1 half of proof generation: given committed instance/advice
/// singles and their per-phase challenges, drives lookups, permutations,
/// vanishing, evaluations, and multiopen.
fn create_proof_from_committed<
    'params,
    'a,
    Scheme: CommitmentScheme,
    P: Prover<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    R: RngCore + 'a,
    T: TranscriptWrite<Scheme::Curve, E>,
>(
    params: &'params Scheme::ParamsProver,
    pk: &ProvingKey<Scheme::Curve>,
    instance: Vec<InstanceSingle<Scheme::Curve>>,
    advice: Vec<AdviceCommitted<Scheme::Curve, LagrangeCoeff>>,
    challenges: Vec<Scheme::Scalar>,
    mut rng: R,
    transcript: &'a mut T,
) -> Result<(), Error>
where
    Scheme::Scalar: Hash + WithSmallOrderMulGroup<3>,
    <Scheme as CommitmentScheme>::ParamsProver: Sync,
{
    let meta = &pk.vk.cs;
    let domain = &pk.vk.domain;

    #[cfg(feature = "profile")]
    let phase2_time = start_timer!(|| "Phase 2: Lookup commit permuted");
    // Sample theta challenge for keeping lookup columns linearly independent
    let theta: ChallengeTheta<_> = transcript.squeeze_challenge_scalar();

    let lookups: Vec<Vec<lookup::prover::Permuted<Scheme::Curve>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Vec<_> {
            // Construct and commit to permuted values for each lookup
            pk.vk
                .cs
                .lookups
                .iter()
                .map(|lookup| {
                    lookup
                        .commit_permuted(
                            pk,
                            params,
                            domain,
                            theta,
                            &advice.advice_polys,
                            &pk.fixed_values,
                            &instance.instance_values,
                            &challenges,
                            &mut rng,
                            transcript,
                        )
                        .unwrap()
                })
                .collect()
        })
        .collect();
    #[cfg(feature = "profile")]
    end_timer!(phase2_time);

    #[cfg(feature = "profile")]
    let phase3a_time = start_timer!(|| "Phase 3a: Commit to permutations");

    // Sample beta challenge
    let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();

    // Sample gamma challenge
    let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();

    // Commit to permutations.
    let permutations: Vec<permutation::prover::Committed<Scheme::Curve>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| {
            pk.vk
                .cs
                .permutation
                .commit(
                    params,
                    pk,
                    &pk.permutation,
                    &advice.advice_polys,
                    &pk.fixed_values,
                    &instance.instance_values,
                    beta,
                    gamma,
                    &mut rng,
                    transcript,
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    #[cfg(feature = "profile")]
    end_timer!(phase3a_time);

    #[cfg(feature = "profile")]
    let phase3b_time = start_timer!(|| "Phase 3b: Lookup commit product");
    let lookups: Vec<Vec<lookup::prover::Committed<Scheme::Curve>>> = lookups
        .into_iter()
        .map(|lookups| -> Vec<_> {
            // Construct and commit to products for each lookup
            lookups
                .into_iter()
                .map(|lookup| {
                    lookup
                        .commit_product(pk, params, beta, gamma, &mut rng, transcript)
                        .unwrap()
                })
                .collect()
        })
        .collect();
    #[cfg(feature = "profile")]
    end_timer!(phase3b_time);

    #[cfg(feature = "profile")]
    let vanishing_time = start_timer!(|| "Commit to vanishing argument's random poly");
    // Commit to the vanishing argument's random polynomial for blinding h(x_3)
    let vanishing = vanishing::Argument::commit(params, domain, &mut rng, transcript).unwrap();

    // Obtain challenge for keeping all separate gates linearly independent
    let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();

    #[cfg(feature = "profile")]
    end_timer!(vanishing_time);
    #[cfg(feature = "profile")]
    let fft_time = start_timer!(|| "Calculate advice polys (fft)");

    // Calculate the advice polys
    let advice: Vec<AdviceCommitted<Scheme::Curve, Coeff>> = advice
        .into_iter()
        .map(
            |AdviceCommitted {
                 advice_polys,
                 advice_blinds,
             }| AdviceCommitted {
                advice_polys: advice_polys
                    .into_iter()
                    .map(|poly| domain.lagrange_to_coeff(poly))
                    .collect::<Vec<_>>(),
                advice_blinds,
            },
        )
        .collect();
    #[cfg(feature = "profile")]
    end_timer!(fft_time);

    #[cfg(feature = "profile")]
    let phase4_time = start_timer!(|| "Phase 4: Evaluate h(X)");
    // Evaluate the h(X) polynomial
    let h_poly = pk.ev.evaluate_h(
        pk,
        &advice
            .iter()
            .map(|a| a.advice_polys.as_slice())
            .collect::<Vec<_>>(),
        &instance
            .iter()
            .map(|i| i.instance_polys.as_slice())
            .collect::<Vec<_>>(),
        &challenges,
        *y,
        *beta,
        *gamma,
        *theta,
        &lookups,
        &permutations,
    );
    #[cfg(feature = "profile")]
    end_timer!(phase4_time);

    #[cfg(feature = "profile")]
    let timer = start_timer!(|| "Commit to vanishing argument's h(X) commitments");
    // Construct the vanishing argument's h(X) commitments
    let vanishing = vanishing.construct(params, domain, h_poly, &mut rng, transcript)?;
    #[cfg(feature = "profile")]
    end_timer!(timer);
    #[cfg(feature = "profile")]
    let eval_time = start_timer!(|| "Commit to vanishing argument's h(X) commitments");

    let x: ChallengeX<_> = transcript.squeeze_challenge_scalar();
    let xn = x.pow([params.n()]);

    if P::QUERY_INSTANCE {
        // Compute and hash instance evals for each circuit instance
        for instance in instance.iter() {
            // Evaluate polynomials at omega^i x
            let instance_evals: Vec<_> = meta
                .instance_queries
                .iter()
                .map(|&(column, at)| {
                    eval_polynomial(
                        &instance.instance_polys[column.index()],
                        domain.rotate_omega(*x, at),
                    )
                })
                .collect();

            // Hash each instance column evaluation
            for eval in instance_evals.iter() {
                transcript.write_scalar(*eval)?;
            }
        }
    }

    // Compute and hash advice evals for each circuit instance
    for advice in advice.iter() {
        // Evaluate polynomials at omega^i x
        let advice_evals: Vec<_> = meta
            .advice_queries
            .iter()
            .map(|&(column, at)| {
                eval_polynomial(
                    &advice.advice_polys[column.index()],
                    domain.rotate_omega(*x, at),
                )
            })
            .collect();

        // Hash each advice column evaluation
        for eval in advice_evals.iter() {
            transcript.write_scalar(*eval)?;
        }
    }

    // Compute and hash fixed evals (shared across all circuit instances)
    let fixed_evals: Vec<_> = meta
        .fixed_queries
        .iter()
        .map(|&(column, at)| {
            eval_polynomial(&pk.fixed_polys[column.index()], domain.rotate_omega(*x, at))
        })
        .collect();

    // Hash each fixed column evaluation
    for eval in fixed_evals.iter() {
        transcript.write_scalar(*eval)?;
    }

    let vanishing = vanishing.evaluate(x, xn, domain, transcript)?;

    // Evaluate common permutation data
    pk.permutation.evaluate(x, transcript)?;

    // Evaluate the permutations, if any, at omega^i x.
    let permutations: Vec<permutation::prover::Evaluated<Scheme::Curve>> = permutations
        .into_iter()
        .map(|permutation| permutation.construct().evaluate(pk, x, transcript).unwrap())
        .collect();

    // Evaluate the lookups, if any, at omega^i x.
    let lookups: Vec<Vec<lookup::prover::Evaluated<Scheme::Curve>>> = lookups
        .into_iter()
        .map(|lookups| -> Vec<_> {
            lookups
                .into_iter()
                .map(|p| p.evaluate(pk, x, transcript).unwrap())
                .collect()
        })
        .collect();
    #[cfg(feature = "profile")]
    end_timer!(eval_time);

    let instances = instance
        .iter()
        .zip(advice.iter())
        .zip(permutations.iter())
        .zip(lookups.iter())
        .flat_map(|(((instance, advice), permutation), lookups)| {
            iter::empty()
                .chain(
                    P::QUERY_INSTANCE
                        .then_some(pk.vk.cs.instance_queries.iter().map(move |&(column, at)| {
                            ProverQuery {
                                point: domain.rotate_omega(*x, at),
                                poly: &instance.instance_polys[column.index()],
                                blind: Blind::default(),
                            }
                        }))
                        .into_iter()
                        .flatten(),
                )
                .chain(
                    pk.vk
                        .cs
                        .advice_queries
                        .iter()
                        .map(move |&(column, at)| ProverQuery {
                            point: domain.rotate_omega(*x, at),
                            poly: &advice.advice_polys[column.index()],
                            blind: advice.advice_blinds[column.index()],
                        }),
                )
                .chain(permutation.open(pk, x))
                .chain(lookups.iter().flat_map(move |p| p.open(pk, x)))
        })
        .chain(
            pk.vk
                .cs
                .fixed_queries
                .iter()
                .map(|&(column, at)| ProverQuery {
                    point: domain.rotate_omega(*x, at),
                    poly: &pk.fixed_polys[column.index()],
                    blind: Blind::default(),
                }),
        )
        .chain(pk.permutation.open(x))
        // We query the h(X) polynomial at x
        .chain(vanishing.open(x));

    #[cfg(feature = "profile")]
    let multiopen_time = start_timer!(|| "Phase 5: multiopen");
    let prover = P::new(params);
    #[allow(clippy::let_and_return)]
    let multiopen_res = prover
        .create_proof(&mut rng, transcript, instances)
        .map_err(|_| Error::ConstraintSystemFailure);
    #[cfg(feature = "profile")]
    end_timer!(multiopen_time);
    multiopen_res
}

#[test]
fn test_create_proof() {
    use crate::{
        circuit::SimpleFloorPlanner,
        plonk::{keygen_pk, keygen_vk, ConstraintSystem},
        poly::kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::ProverSHPLONK,
        },
        transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
    };
    use halo2curves::bn256::Bn256;
    use rand_core::OsRng;

    #[derive(Clone, Copy)]
    struct MyCircuit;

    impl<F: Field> Circuit<F> for MyCircuit {
        type Config = ();
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            *self
        }

        fn configure(_meta: &mut ConstraintSystem<F>) -> Self::Config {}

        fn synthesize(
            &self,
            _config: Self::Config,
            _layouter: impl crate::circuit::Layouter<F>,
        ) -> Result<(), Error> {
            Ok(())
        }
    }

    let params: ParamsKZG<Bn256> = ParamsKZG::setup(3, OsRng);
    let vk = keygen_vk(&params, &MyCircuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &MyCircuit).expect("keygen_pk should not fail");
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

    // Create proof with wrong number of instances
    let proof = create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[MyCircuit, MyCircuit],
        &[],
        OsRng,
        &mut transcript,
    );
    assert!(matches!(proof.unwrap_err(), Error::InvalidInstances));

    // Create proof with correct number of instances
    create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[MyCircuit, MyCircuit],
        &[&[], &[]],
        OsRng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
}
