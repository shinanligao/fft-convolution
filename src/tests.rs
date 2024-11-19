#[cfg(test)]
mod tests {
    use crate::crossfade_convolver_fd::CrossfadeConvolverFrequencyDomain;
    use crate::crossfade_convolver_td::CrossfadeConvolverTimeDomain;
    use crate::fft_convolver::{FFTConvolverOLA, FFTConvolverOLS};
    use crate::{Convolution, Sample};

    fn generate_sinusoid(
        length: usize,
        frequency: f32,
        sample_rate: f32,
        gain: f32,
    ) -> Vec<Sample> {
        let mut signal = vec![0.0; length];
        for i in 0..length {
            signal[i] =
                gain * (2.0 * std::f32::consts::PI * frequency * i as Sample / sample_rate).sin();
        }
        signal
    }

    #[test]
    fn ola_and_ols_are_equivalent() {
        let block_size = 512;
        let response = generate_sinusoid(8 * block_size, 1000.0, 48000.0, 1.0);
        let num_input_blocks = 16;
        let input = generate_sinusoid(num_input_blocks * block_size, 1300.0, 48000.0, 1.0);
        let mut output_ola = vec![0.0; num_input_blocks * block_size];
        let mut output_ols = vec![0.0; num_input_blocks * block_size];
        for block_size_factor in [0.5, 0.7, 1.0, 1.8, 2.0] {
            let convolver_block_size = (block_size as f32 * block_size_factor) as usize;
            let mut convolver_ola =
                FFTConvolverOLA::init(&response, convolver_block_size, response.len());
            let mut convolver_ols =
                FFTConvolverOLS::init(&response, convolver_block_size, response.len());

            for i in 0..num_input_blocks {
                convolver_ola.process(
                    &input[i * block_size..(i + 1) * block_size],
                    &mut output_ola[i * block_size..(i + 1) * block_size],
                );
                convolver_ols.process(
                    &input[i * block_size..(i + 1) * block_size],
                    &mut output_ols[i * block_size..(i + 1) * block_size],
                );
                for j in i * block_size..(i + 1) * block_size {
                    assert!((output_ola[j] - output_ols[j]).abs() < 1e-4);
                }
            }
        }
    }

    #[test]
    fn fft_convolver_update_is_reset() {
        let block_size = 512;
        let response_a = generate_sinusoid(block_size, 1000.0, 48000.0, 1.0);
        let response_b = generate_sinusoid(block_size, 2000.0, 48000.0, 0.7);
        let mut convolver_a = FFTConvolverOLA::init(&response_a, block_size, response_a.len());
        let mut convolver_b = FFTConvolverOLA::init(&response_b, block_size, response_b.len());
        let mut convolver_update = FFTConvolverOLA::init(&response_a, block_size, response_a.len());
        let mut output_a = vec![0.0; block_size];
        let mut output_b = vec![0.0; block_size];
        let mut output_update = vec![0.0; block_size];

        let num_input_blocks = 16;
        let input = generate_sinusoid(num_input_blocks * block_size, 1300.0, 48000.0, 1.0);

        let update_index = 8;

        for i in 0..num_input_blocks {
            if i == update_index {
                convolver_update.update(&response_b);
            }

            convolver_update.process(
                &input[i * block_size..(i + 1) * block_size],
                &mut output_update,
            );

            let check_equal = |lhs: &[Sample], rhs: &[Sample]| {
                for j in 0..block_size {
                    assert!((lhs[j] - rhs[j]).abs() < 1e-6);
                }
            };

            if i < update_index {
                convolver_a.process(&input[i * block_size..(i + 1) * block_size], &mut output_a);
                check_equal(&output_a, &output_update);
            } else {
                convolver_b.process(&input[i * block_size..(i + 1) * block_size], &mut output_b);
                check_equal(&output_b, &output_update);
            }
        }
    }

    #[test]
    fn test_crossfade_convolver_td() {
        let block_size = 512;
        let response_a = generate_sinusoid(block_size, 1000.0, 48000.0, 1.0);
        let response_b = generate_sinusoid(block_size, 2000.0, 48000.0, 0.7);

        fn test_convolver_type<C>(
            response_a: &[Sample],
            response_b: &[Sample],
            block_size: usize,
        ) -> Vec<Sample>
        where
            C: Convolution,
        {
            let mut convolver_a = C::init(response_a, block_size, response_a.len());
            let mut convolver_b = C::init(response_b, block_size, response_b.len());
            let mut crossfade_convolver = CrossfadeConvolverTimeDomain::new(
                convolver_a.clone(),
                block_size,
                block_size,
                block_size,
            );
            let mut output_a = vec![0.0; block_size];
            let mut output_b = vec![0.0; block_size];
            let mut output_crossfade_convolver = vec![0.0; block_size];

            let num_input_blocks = 16;
            let input = generate_sinusoid(num_input_blocks * block_size, 1300.0, 48000.0, 1.0);
            let update_index = 8;

            for i in 0..num_input_blocks {
                if i == update_index {
                    crossfade_convolver.update(&response_b);
                }

                crossfade_convolver.process(
                    &input[i * block_size..(i + 1) * block_size],
                    &mut output_crossfade_convolver,
                );

                let check_equal = |lhs: &[Sample], rhs: &[Sample]| {
                    for j in 0..block_size {
                        assert!((lhs[j] - rhs[j]).abs() < 1e-6);
                    }
                };

                convolver_a.process(&input[i * block_size..(i + 1) * block_size], &mut output_a);
                if i >= update_index {
                    convolver_b
                        .process(&input[i * block_size..(i + 1) * block_size], &mut output_b);
                }

                if i <= update_index {
                    check_equal(&output_a, &output_crossfade_convolver);
                } else {
                    if i == update_index + 1 {
                        // crossover sample
                        let crossover_index = block_size / 2 - 1;
                        assert!(
                            (output_crossfade_convolver[crossover_index]
                                - (output_a[crossover_index] * 0.5
                                    + output_b[crossover_index] * 0.5))
                                .abs()
                                < 1e-6
                        );
                    } else {
                        check_equal(&output_b, &output_crossfade_convolver);
                    }
                }
            }
            output_crossfade_convolver
        }

        let ola_outputs =
            test_convolver_type::<FFTConvolverOLA>(&response_a, &response_b, block_size);
        let ols_outputs =
            test_convolver_type::<FFTConvolverOLS>(&response_a, &response_b, block_size);

        for (sample_ola, sample_ols) in ola_outputs.iter().zip(ols_outputs.iter()) {
            assert!((sample_ola - sample_ols).abs() < 1e-4);
        }
    }

    #[test]
    fn time_domain_and_frequency_crossfaders_are_equivalent() {
        let block_size = 512;
        let response_a = generate_sinusoid(block_size * 4, 1000.0, 48000.0, 1.0);
        let response_b = generate_sinusoid(block_size * 4, 2000.0, 48000.0, 0.7);
        let num_input_blocks = 16;
        let input = generate_sinusoid(num_input_blocks * block_size, 1300.0, 48000.0, 1.0);
        let mut output_td = vec![0.0; num_input_blocks * block_size];
        let mut output_fd = vec![0.0; num_input_blocks * block_size];
        let mut crossfade_convolver_td = CrossfadeConvolverTimeDomain::<FFTConvolverOLS>::new(
            FFTConvolverOLS::init(&response_a, block_size, response_a.len()),
            response_a.len(),
            block_size,
            block_size,
        );
        let mut crossfade_convolver_fd =
            CrossfadeConvolverFrequencyDomain::init(&response_a, block_size, response_a.len());

        let update_index = 8;
        for i in 0..num_input_blocks {
            if i == update_index {
                crossfade_convolver_td.update(&response_b);
            }

            if i == update_index + 1 {
                crossfade_convolver_fd.update(&response_b);
            }

            crossfade_convolver_td.process(
                &input[i * block_size..(i + 1) * block_size],
                &mut output_td[i * block_size..(i + 1) * block_size],
            );
            crossfade_convolver_fd.process(
                &input[i * block_size..(i + 1) * block_size],
                &mut output_fd[i * block_size..(i + 1) * block_size],
            );
            for j in i * block_size..(i + 1) * block_size {
                assert!((output_td[j] - output_fd[j]).abs() < 1e-1); // TODO: align start of crossfade between TD and FD, which should reduce this error
            }
        }
    }
}
