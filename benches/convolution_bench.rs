use convolution::{
    crossfade_convolver_fd::CrossfadeConvolverFrequencyDomain,
    crossfade_convolver_td::CrossfadeConvolverTimeDomain,
    faded_stepwise_update_convolver::FadedStepwiseUpdateConvolver, fft_convolver::FFTConvolverOLS,
    stepwise_update_convolver::StepwiseUpdateConvolver,
};
use convolution::{Convolution, Sample};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::f64::consts::PI;

const MAX_FILTER_LENGTH: usize = 16384;
const BLOCK_SIZES: [usize; 3] = [128, 512, 2048];
const TRANSITION_SAMPLES: usize = 1048576 + 10 * 2048;

pub fn convolver_update_process_benchmarks(c: &mut Criterion) {
    let amplitude = 10.0;
    let f_s = 48000.0;

    let sinusoid = |frequency: Sample, amplitude: Sample, length: usize| {
        (0..length)
            .map(|i| amplitude * (frequency / f_s * 2.0 * PI as Sample * i as Sample).sin())
            .collect::<Vec<Sample>>()
    };

    let mut group = c.benchmark_group("convolver_update_process");

    let max_block_size = *BLOCK_SIZES.iter().max().unwrap();
    let input_long: Vec<Sample> = sinusoid(440.0, amplitude, max_block_size);

    for block_size in BLOCK_SIZES.iter() {
        let input = &input_long[0..*block_size];
        let mut output_setup = vec![0.0; *block_size];
        let mut output_benchmark = vec![0.0; *block_size];

        let mut ir_len = 0;

        while ir_len < MAX_FILTER_LENGTH {
            ir_len += block_size;
            let response: Vec<Sample> = sinusoid(100.0, 1.0, ir_len);

            group.bench_with_input(
                BenchmarkId::new(
                    "time_domain_crossfade",
                    format!("b{}_ir{}", block_size, ir_len),
                ),
                &ir_len,
                |b, _ir_len| {
                    b.iter_batched_ref(
                        || -> CrossfadeConvolverTimeDomain<FFTConvolverOLS> {
                            let mut convolver =
                                CrossfadeConvolverTimeDomain::<FFTConvolverOLS>::new(
                                    FFTConvolverOLS::init(&response, *block_size, response.len()),
                                    response.len(),
                                    *block_size,
                                    TRANSITION_SAMPLES,
                                );
                            convolver.process(&input, &mut output_setup);
                            convolver
                        },
                        |convolver: &mut CrossfadeConvolverTimeDomain<FFTConvolverOLS>| {
                            convolver.update(&response);
                            convolver.process(&input, &mut output_benchmark);
                        },
                        criterion::BatchSize::SmallInput,
                    )
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    "frequency_domain_crossfade",
                    format!("b{}_ir{}", block_size, ir_len),
                ),
                &ir_len,
                |b, _ir_len| {
                    b.iter_batched_ref(
                        || -> CrossfadeConvolverFrequencyDomain {
                            let mut convolver = CrossfadeConvolverFrequencyDomain::init(
                                &response,
                                *block_size,
                                response.len(),
                            );
                            convolver.process(&input, &mut output_setup);
                            convolver
                        },
                        |convolver: &mut CrossfadeConvolverFrequencyDomain| {
                            convolver.update(&response);
                            convolver.process(&input, &mut output_benchmark);
                        },
                        criterion::BatchSize::SmallInput,
                    )
                },
            );

            group.bench_with_input(
                BenchmarkId::new("stepwise_update", format!("b{}_ir{}", block_size, ir_len)),
                &ir_len,
                |b, _ir_len| {
                    b.iter_batched_ref(
                        || -> StepwiseUpdateConvolver {
                            let mut convolver = StepwiseUpdateConvolver::new(
                                &response,
                                response.len(),
                                *block_size,
                            );
                            convolver.process(&input, &mut output_setup);
                            convolver
                        },
                        |convolver: &mut StepwiseUpdateConvolver| {
                            convolver.update(&response);
                            convolver.process(&input, &mut output_benchmark);
                        },
                        criterion::BatchSize::SmallInput,
                    )
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    "faded_stepwise_update",
                    format!("b{}_ir{}", block_size, ir_len),
                ),
                &ir_len,
                |b, _ir_len| {
                    b.iter_batched_ref(
                        || -> FadedStepwiseUpdateConvolver {
                            let mut convolver = FadedStepwiseUpdateConvolver::new(
                                &response,
                                response.len(),
                                *block_size,
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
    }
    group.finish();
}

criterion_group!(benches, convolver_update_process_benchmarks);
criterion_main!(benches);
