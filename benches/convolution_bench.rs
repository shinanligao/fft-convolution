use convolution::{
    crossfade_convolver_fd::CrossfadeConvolverFrequencyDomain,
    crossfade_convolver_td::CrossfadeConvolverTimeDomain,
    faded_stepwise_update_convolver::FadedStepwiseUpdateConvolver, fft_convolver::FFTConvolverOLS,
    stepwise_update_convolver::StepwiseUpdateConvolver,
};
use convolution::{Convolution, Sample};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::f64::consts::PI;

const BLOCK_SIZE: usize = 256;
const FILTER_LENGTHS: [usize; 5] = [128, 256, 512, 1024, 2048];
const TRANSITION_SAMPLES: usize = 12000;

pub fn convolver_update_process_benchmarks(c: &mut Criterion) {
    let amplitude = 10.0;

    let sinusoid = |frequency: Sample, amplitude: Sample, length: usize| {
        (0..length)
            .map(|i| amplitude * (frequency * 2.0 * PI as Sample * i as Sample).sin())
            .collect::<Vec<Sample>>()
    };

    let mut group = c.benchmark_group("convolver_update_process");

    let input: Vec<Sample> = sinusoid(440.0, amplitude, BLOCK_SIZE);
    let mut output_setup = vec![0.0; BLOCK_SIZE];
    let mut output_benchmark = vec![0.0; BLOCK_SIZE];

    for ir_len in FILTER_LENGTHS.iter() {
        let response: Vec<Sample> = sinusoid(100.0, 1.0, *ir_len);

        // group.bench_with_input(
        //     BenchmarkId::new("time_domain_crossfade", ir_len),
        //     ir_len,
        //     |b, _ir_len| {
        //         b.iter_batched_ref(
        //             || -> CrossfadeConvolverTimeDomain<FFTConvolverOLS> {
        //                 let mut convolver = CrossfadeConvolverTimeDomain::<FFTConvolverOLS>::new(
        //                     FFTConvolverOLS::init(&response, BLOCK_SIZE, response.len()),
        //                     response.len(),
        //                     BLOCK_SIZE,
        //                     TRANSITION_SAMPLES,
        //                 );
        //                 convolver.process(&input, &mut output_setup);
        //                 convolver
        //             },
        //             |convolver: &mut CrossfadeConvolverTimeDomain<FFTConvolverOLS>| {
        //                 convolver.update(&response);
        //                 convolver.process(&input, &mut output_benchmark);
        //             },
        //             criterion::BatchSize::SmallInput,
        //         )
        //     },
        // );

        // group.bench_with_input(
        //     BenchmarkId::new("frequency_domain_crossfade", ir_len),
        //     ir_len,
        //     |b, _ir_len| {
        //         b.iter_batched_ref(
        //             || -> CrossfadeConvolverFrequencyDomain {
        //                 let mut convolver = CrossfadeConvolverFrequencyDomain::init(
        //                     &response,
        //                     BLOCK_SIZE,
        //                     response.len(),
        //                 );
        //                 convolver.process(&input, &mut output_setup);
        //                 convolver
        //             },
        //             {
        //                 |convolver: &mut CrossfadeConvolverFrequencyDomain| {
        //                     convolver.update(&response);
        //                     convolver.process(&input, &mut output_benchmark);
        //                 }
        //             },
        //             criterion::BatchSize::SmallInput,
        //         )
        //     },
        // );

        // group.bench_with_input(
        //     BenchmarkId::new("stepwise_update", ir_len),
        //     ir_len,
        //     |b, _ir_len| {
        //         b.iter_batched_ref(
        //             || -> StepwiseUpdateConvolver {
        //                 let mut convolver =
        //                     StepwiseUpdateConvolver::new(&response, response.len(), BLOCK_SIZE);
        //                 convolver.process(&input, &mut output_setup);
        //                 convolver
        //             },
        //             |convolver: &mut StepwiseUpdateConvolver| {
        //                 convolver.update(&response);
        //                 convolver.process(&input, &mut output_benchmark);
        //             },
        //             criterion::BatchSize::SmallInput,
        //         )
        //     },
        // );

        group.bench_with_input(
            BenchmarkId::new("faded_stepwise_update", ir_len),
            ir_len,
            |b, _ir_len| {
                b.iter_batched_ref(
                    || -> FadedStepwiseUpdateConvolver {
                        let mut convolver = FadedStepwiseUpdateConvolver::new(
                            &response,
                            response.len(),
                            BLOCK_SIZE,
                            TRANSITION_SAMPLES,
                        );
                        convolver.process(&input, &mut output_setup);
                        convolver
                    },
                    |convolver: &mut FadedStepwiseUpdateConvolver| {
                        convolver.update(&response);
                        convolver.process(&input, &mut output_benchmark);
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

criterion_group!(benches, convolver_update_process_benchmarks);
criterion_main!(benches);
