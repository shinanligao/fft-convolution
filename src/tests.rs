#[cfg(test)]
pub mod tests {
    use crate::crossfade_convolver_fd::CrossfadeConvolverFrequencyDomain;
    use crate::crossfade_convolver_td::CrossfadeConvolverTimeDomain;
    use crate::faded_stepwise_update_convolver::FadedStepwiseUpdateConvolver;
    use crate::fft_convolver::{FFTConvolverOLA, FFTConvolverOLS};
    use crate::stepwise_update_convolver::StepwiseUpdateConvolver;
    use crate::{Convolution, Sample};

    use serde_json::Value;
    use std::cmp::Ordering;
    use std::fs::File;
    use std::io::{BufRead, BufReader, Error};

    pub fn generate_sinusoid(
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

    fn sort_by_metric(json_obj: &Value) -> Vec<(String, String, String, f64)> {
        let mut results = Vec::new();

        if let Value::Object(distance_map) = json_obj {
            for (distance, block_sizes) in distance_map {
                if let Value::Object(block_size_map) = block_sizes {
                    for (block_size, algorithms) in block_size_map {
                        if let Value::Object(algorithm_map) = algorithms {
                            for (algorithm, metric_value) in algorithm_map {
                                if let Value::Number(metric) = metric_value {
                                    if let Some(metric_f64) = metric.as_f64() {
                                        results.push((
                                            distance.clone(),
                                            block_size.clone(),
                                            algorithm.clone(),
                                            metric_f64,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        results.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(Ordering::Equal));

        results
    }

    fn read_vector_from_file(file_path: &str) -> Result<Vec<Sample>, Error> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut vec = Vec::new();

        for line in reader.lines() {
            let line = line?;
            // Parse each line as a Sample and collect results
            match line.trim().parse::<Sample>() {
                Ok(value) => vec.push(value),
                Err(_) => {
                    return Err(Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Failed to parse float",
                    ))
                }
            }
        }

        Ok(vec)
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

    #[test]
    fn stepwise_update_is_like_zero_crossfade() {
        let block_size = 256;
        let num_segments = 32;
        let response_a = generate_sinusoid(num_segments * block_size, 500.0, 48000.0, 0.5);
        let response_b = generate_sinusoid(num_segments * block_size, 400.0, 48000.0, 0.9);
        let mut convolver_a = FFTConvolverOLA::init(&response_a, block_size, response_a.len());
        let mut convolver_b = FFTConvolverOLA::init(&response_b, block_size, response_b.len());
        let mut stepwise_update_convolver =
            StepwiseUpdateConvolver::init(&response_a, block_size, response_a.len());
        let mut input_gain_a = 1.0;
        let mut input_gain_b = 0.0;

        let num_input_blocks = num_segments * 4;
        let input = generate_sinusoid(num_input_blocks * block_size, 200.0, 48000.0, 0.3);

        let mut output_a = vec![0.0; num_input_blocks * block_size];
        let mut output_b = vec![0.0; num_input_blocks * block_size];
        let mut output_stepwise_update_convolver = vec![0.0; num_input_blocks * block_size];

        let update_index = num_segments * 2;

        for i in 0..num_input_blocks {
            if i == update_index {
                input_gain_a = 0.0;
                input_gain_b = 1.0;
                stepwise_update_convolver.update(&response_b);
            }

            convolver_a.process(
                &input
                    .iter()
                    .map(|&x| x * input_gain_a)
                    .collect::<Vec<Sample>>()[i * block_size..(i + 1) * block_size],
                &mut output_a[i * block_size..(i + 1) * block_size],
            );

            convolver_b.process(
                &input
                    .iter()
                    .map(|&x| x * input_gain_b)
                    .collect::<Vec<Sample>>()[i * block_size..(i + 1) * block_size],
                &mut output_b[i * block_size..(i + 1) * block_size],
            );

            stepwise_update_convolver.process(
                &input[i * block_size..(i + 1) * block_size],
                &mut output_stepwise_update_convolver[i * block_size..(i + 1) * block_size],
            );
        }

        let check_equal = |lhs: &[Sample], rhs: &[Sample]| {
            for j in 0..lhs.len() {
                assert!((lhs[j] - rhs[j]).abs() < 1e-4);
            }
        };

        check_equal(
            &output_a
                .iter()
                .zip(output_b.iter())
                .map(|(a, b)| a + b)
                .collect::<Vec<Sample>>(),
            &output_stepwise_update_convolver,
        );
    }

    #[test]
    fn faded_stepwise_update_corresponds_to_input_fading() {
        let block_size = 256;
        let num_segments = 32;
        let response_a = generate_sinusoid(num_segments * block_size, 500.0, 48000.0, 0.5);
        let response_b = generate_sinusoid(num_segments * block_size, 400.0, 48000.0, 0.9);

        let num_input_blocks = num_segments * 8;
        let transition_samples = num_segments * block_size * 3;

        for block_factor in [0.5, 1.0, 2.0] {
            let outer_block_size = (block_size as f32 / block_factor) as usize;

            let mut convolver_a = FFTConvolverOLA::init(&response_a, block_size, response_a.len());
            let mut convolver_b = FFTConvolverOLA::init(&response_b, block_size, response_b.len());

            let fade_steps = ((transition_samples - response_a.len()) / block_size) as usize;

            let mut faded_stepwise_update_convolver = FadedStepwiseUpdateConvolver::new(
                &response_a,
                response_a.len(),
                block_size,
                transition_samples,
            );

            let input = generate_sinusoid(num_input_blocks * block_size, 100.0, 48000.0, 0.3);
            let mut input_gains_a = vec![1.0; num_input_blocks * block_size];
            let mut input_gains_b = vec![0.0; num_input_blocks * block_size];

            let mut output_a = vec![0.0; num_input_blocks * block_size];
            let mut output_b = vec![0.0; num_input_blocks * block_size];
            let mut output_faded_stepwise_update_convolver =
                vec![0.0; num_input_blocks * block_size];

            let num_outer_blocks = (num_input_blocks as f32 * block_factor) as usize;
            let update_index = (num_segments as f32 * 2.0 * block_factor) as usize;

            for i in 0..num_input_blocks * block_size {
                if i < update_index * outer_block_size {
                    continue;
                }
                let step = (i - update_index * outer_block_size) / block_size + 1;
                let weight = (step as f32 / fade_steps as f32).min(1.0);
                input_gains_a[i] = 1.0 - weight;
                input_gains_b[i] = weight;
            }

            for i in 0..num_outer_blocks {
                if i == update_index {
                    faded_stepwise_update_convolver.update(&response_b);
                }

                convolver_a.process(
                    &input
                        .iter()
                        .enumerate()
                        .map(|(j, &x)| x * input_gains_a[j])
                        .collect::<Vec<Sample>>()[i * outer_block_size..(i + 1) * outer_block_size],
                    &mut output_a[i * outer_block_size..(i + 1) * outer_block_size],
                );

                convolver_b.process(
                    &input
                        .iter()
                        .enumerate()
                        .map(|(j, &x)| x * input_gains_b[j])
                        .collect::<Vec<Sample>>()[i * outer_block_size..(i + 1) * outer_block_size],
                    &mut output_b[i * outer_block_size..(i + 1) * outer_block_size],
                );

                faded_stepwise_update_convolver.process(
                    &input[i * outer_block_size..(i + 1) * outer_block_size],
                    &mut output_faded_stepwise_update_convolver
                        [i * outer_block_size..(i + 1) * outer_block_size],
                );
            }

            let check_equal = |lhs: &[Sample], rhs: &[Sample]| {
                for j in 0..lhs.len() {
                    assert!((lhs[j] - rhs[j]).abs() < 1e-4);
                }
            };

            check_equal(
                &output_a
                    .iter()
                    .zip(output_b.iter())
                    .map(|(a, b)| a + b)
                    .collect::<Vec<Sample>>(),
                &output_faded_stepwise_update_convolver,
            );
        }
    }

    #[test]
    fn faded_stepwise_update_convolver_independent_from_outer_block_size() {
        let block_size = 256;
        let num_segments = 32;
        let num_input_blocks = num_segments * 8;
        let transition_samples = num_segments * block_size * 3;

        let mut outputs: Vec<Vec<Sample>> = Vec::new();

        let block_factors = [0.5, 1.0, 2.0];

        for block_factor in block_factors {
            let outer_block_size = (block_size as f32 / block_factor) as usize;
            let response_a = generate_sinusoid(num_segments * block_size, 500.0, 48000.0, 0.5);
            let response_b = generate_sinusoid(num_segments * block_size, 400.0, 48000.0, 0.9);

            let mut faded_stepwise_update_convolver = FadedStepwiseUpdateConvolver::new(
                &response_a,
                response_a.len(),
                block_size,
                transition_samples,
            );

            let input = generate_sinusoid(num_input_blocks * block_size, 100.0, 48000.0, 0.3);

            let mut output_faded_stepwise_update_convolver =
                vec![0.0; num_input_blocks * block_size];

            let num_outer_blocks = (num_input_blocks as f32 * block_factor) as usize;
            let update_index = (num_segments as f32 * 2.0 * block_factor) as usize;

            for i in 0..num_outer_blocks {
                if i == update_index {
                    faded_stepwise_update_convolver.update(&response_b);
                }

                faded_stepwise_update_convolver.process(
                    &input[i * outer_block_size..(i + 1) * outer_block_size],
                    &mut output_faded_stepwise_update_convolver
                        [i * outer_block_size..(i + 1) * outer_block_size],
                );
            }

            outputs.push(output_faded_stepwise_update_convolver);
        }

        for i in 0..block_factors.len() - 1 {
            let check_equal = |lhs: &[Sample], rhs: &[Sample]| {
                for j in 0..lhs.len() {
                    assert!((lhs[j] - rhs[j]).abs() < 1e-4);
                }
            };

            check_equal(&outputs[i], &outputs[i + 1]);
        }
    }

    #[test]
    fn compare_sideband_energy() {
        use crate::evaluation::sideband_energy::sideband_energy;
        use serde_json::json;
        use std::fs::File;
        use std::io::Write;

        let frequency = 100.0;
        let sample_rate = 48000.0;

        fn run_convolver<C>(
            response_b: &[Sample],
            update_index: usize,
            num_input_blocks: usize,
            convolver: &mut C,
            block_size: usize,
            frequency: Sample,
            sample_rate: Sample,
        ) -> Vec<Sample>
        where
            C: Convolution,
        {
            let mut output_fd = vec![0.0; num_input_blocks * block_size];

            let input =
                generate_sinusoid(num_input_blocks * block_size, frequency, sample_rate, 1.0);

            for i in 0..num_input_blocks {
                if i == update_index {
                    convolver.update(&response_b);
                }

                convolver.process(
                    &input[i * block_size..(i + 1) * block_size],
                    &mut output_fd[i * block_size..(i + 1) * block_size],
                );
            }
            output_fd
        }

        let target_transition_duration: usize = 12000;

        let mut results = json!({});

        for distance in [50, 100, 200] {
            let mut results_distance = json!({});

            let response_a =
                read_vector_from_file(&format!("resources/response_0_{}.txt", distance)).unwrap();
            let response_b =
                read_vector_from_file(&format!("resources/response_1_{}.txt", distance)).unwrap();

            let max_response_length = response_a.len().max(response_b.len());

            for block_size in [128, 512, 2048] {
                // ensure that transition is started after the transient ramp-up is complete
                let update_index = response_a.len().div_ceil(block_size);

                let num_input_blocks =
                    update_index + target_transition_duration.div_ceil(block_size);

                // Time Domain Crossfade Convolver
                let mut crossfade_convolver_td =
                    CrossfadeConvolverTimeDomain::<FFTConvolverOLS>::new(
                        FFTConvolverOLS::init(&response_a, block_size, max_response_length),
                        response_a.len(),
                        block_size,
                        target_transition_duration,
                    );

                let mut crossfade_convolver_fd = CrossfadeConvolverFrequencyDomain::init(
                    &response_a,
                    block_size,
                    max_response_length,
                );

                let mut stepwise_update_convolver =
                    StepwiseUpdateConvolver::init(&response_a, block_size, max_response_length);

                let mut faded_stepwise_update_convolver = FadedStepwiseUpdateConvolver::new(
                    &response_a,
                    max_response_length,
                    block_size,
                    target_transition_duration,
                );

                let output_td = run_convolver(
                    &response_b,
                    update_index,
                    num_input_blocks,
                    &mut crossfade_convolver_td,
                    block_size,
                    frequency,
                    sample_rate,
                );

                let output_fd = run_convolver(
                    &response_b,
                    update_index,
                    num_input_blocks,
                    &mut crossfade_convolver_fd,
                    block_size,
                    frequency,
                    sample_rate,
                );

                let output_stepwise = run_convolver(
                    &response_b,
                    update_index,
                    num_input_blocks,
                    &mut stepwise_update_convolver,
                    block_size,
                    frequency,
                    sample_rate,
                );

                let output_faded = run_convolver(
                    &response_b,
                    update_index,
                    num_input_blocks,
                    &mut faded_stepwise_update_convolver,
                    block_size,
                    frequency,
                    sample_rate,
                );

                let sideband_energy_td = sideband_energy(
                    &output_td[update_index * block_size
                        ..update_index * block_size + target_transition_duration],
                    frequency,
                    sample_rate,
                );

                let sideband_energy_fd = sideband_energy(
                    &output_fd[update_index * block_size
                        ..update_index * block_size + target_transition_duration],
                    frequency,
                    sample_rate,
                );

                let sideband_energy_stepwise = sideband_energy(
                    &output_stepwise[update_index * block_size
                        ..update_index * block_size + target_transition_duration],
                    frequency,
                    sample_rate,
                );

                let sideband_energy_faded = sideband_energy(
                    &output_faded[update_index * block_size
                        ..update_index * block_size + target_transition_duration],
                    frequency,
                    sample_rate,
                );

                let key_td = "time_domain_crossfade";
                let key_fd = "frequency_domain_crossfade";
                let key_step = "stepwise_update";
                let key_faded = "faded_stepwise_update";

                let results_block = json!({
                    key_td: sideband_energy_td,
                    key_fd: sideband_energy_fd,
                    key_step: sideband_energy_stepwise,
                    key_faded: sideband_energy_faded,
                });

                results_distance[block_size.to_string()] = results_block;
            }
            results[distance.to_string()] = results_distance;
        }

        let mut file = File::create("sideband_energy.json").unwrap();
        file.write_all(serde_json::to_string_pretty(&results).unwrap().as_bytes())
            .unwrap();

        let sorted_results = sort_by_metric(&results);

        for (distance, block_size, algorithm, metric) in sorted_results {
            println!(
                "Distance: {}, Block Size: {}, Algorithm: {}, Metric: {}",
                distance, block_size, algorithm, metric
            );
        }
    }
}
