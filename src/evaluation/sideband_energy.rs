use crate::fft_convolver::{complex_size, Fft};
use crate::Sample;
use rustfft::num_complex::Complex;

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

fn hann_window(size: usize) -> Vec<Sample> {
    (0..size)
        .map(|i| {
            let x = 2.0 * std::f64::consts::PI as Sample * i as Sample / (size - 1) as Sample;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

pub fn sideband_energy(input: &[Sample], input_frequency: Sample, sample_rate: Sample) -> Sample {
    let buffer_size = input.len();
    let mut fft = Fft::default();
    fft.init(buffer_size);

    let window = hann_window(buffer_size);
    let mut fft_buffer = input
        .iter()
        .zip(&window)
        .map(|(s, w)| s * w)
        .collect::<Vec<Sample>>();
    let coherent_gain: Sample = window.iter().sum::<Sample>() / window.len() as Sample;

    let fft_complex_size = complex_size(buffer_size);
    let mut spectrum = vec![Complex::new(0., 0.); fft_complex_size];
    fft.forward(&mut fft_buffer, &mut spectrum).unwrap();

    let frequency_bin = |input_frequency: Sample| -> usize {
        (input_frequency * buffer_size as f32 / sample_rate).round() as usize
    };

    let erb = 6.23 * (input_frequency / 1000.0) * (input_frequency / 1000.0)
        + 93.39 * (input_frequency / 1000.0)
        + 28.52;

    let upper_bound = frequency_bin(input_frequency + erb / 2.);
    let lower_bound = frequency_bin(input_frequency - erb / 2.);

    let magnitude: Vec<Sample> = spectrum
        .iter()
        .map(|x| 2.0 * x.norm() / (coherent_gain * buffer_size as Sample))
        .collect();

    let sum_squared: Sample = magnitude
        .iter()
        .enumerate()
        .filter(|&(i, _)| i < lower_bound || i > upper_bound)
        .map(|(_, c)| c)
        .sum();

    return (sum_squared / fft_complex_size as Sample).sqrt();
}

#[test]
fn test_sideband_energy() {
    let num_samples = 4080;
    let sample_rate = 48000.0;
    let frequency = 100.0;
    let clean_signal = generate_sinusoid(num_samples, frequency, sample_rate, 1.0);

    assert!(sideband_energy(&clean_signal, frequency, sample_rate) < 0.01);

    let dirty_signal = clean_signal
        .iter()
        .zip(generate_sinusoid(num_samples, frequency * 8.0, sample_rate, 1.0).iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<Sample>>();

    assert!(sideband_energy(&dirty_signal, frequency, sample_rate) > 0.01);
}
