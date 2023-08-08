// todo: use a generic floating point type
type Sample = f32;

pub trait Convolution {
    fn init(response: &[Sample], max_block_size: usize) -> Self;
    fn update(&mut self, response: &[Sample]);
    fn process(&mut self, input: &[Sample], output: &mut [Sample]);
}

pub struct Convolver<Processor: Convolution> {
    processor: Processor,
}

impl<Processor: Convolution> Convolver<Processor> {
    pub fn new(convolver_processor: Processor) -> Self {
        Self {
            processor: convolver_processor,
        }
    }

    pub fn set_response(&mut self, response: &[Sample]) {
        self.processor.update(response);
    }

    pub fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        self.processor.process(input, output);
    }
}

#[test]
fn test_fft_convolver() {
    let mut response = [0.0; 1024];
    response[0] = 1.0;
    let mut convolver = Convolver::new(FFTConvolver::init(&response, 1024));
    let input = vec![1.0; 1024];
    let mut output = vec![0.0; 1024];
    convolver.process(&input, &mut output);

    for i in 0..1024 {
        assert!((output[i] - 1.0).abs() < 1e-6);
    }
}

// ------ FFTConvolver ------

use realfft::{ComplexToReal, FftError, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

pub struct Fft {
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
}

impl std::fmt::Debug for Fft {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

impl Fft {
    pub fn default() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        Self {
            fft_forward: planner.plan_fft_forward(0),
            fft_inverse: planner.plan_fft_inverse(0),
        }
    }

    pub fn init(&mut self, length: usize) {
        let mut planner = RealFftPlanner::<f32>::new();
        self.fft_forward = planner.plan_fft_forward(length);
        self.fft_inverse = planner.plan_fft_inverse(length);
    }

    pub fn forward(&self, input: &mut [f32], output: &mut [Complex<f32>]) -> Result<(), FftError> {
        self.fft_forward.process(input, output)?;
        Ok(())
    }

    pub fn inverse(&self, input: &mut [Complex<f32>], output: &mut [f32]) -> Result<(), FftError> {
        self.fft_inverse.process(input, output)?;

        // FFT Normalization
        let len = output.len();
        output.iter_mut().for_each(|bin| *bin /= len as f32);

        Ok(())
    }
}

pub fn complex_size(size: usize) -> usize {
    (size / 2) + 1
}

pub fn copy_and_pad(dst: &mut [f32], src: &[f32], src_size: usize) {
    assert!(dst.len() >= src_size);
    dst[0..src_size].clone_from_slice(&src[0..src_size]);
    dst[src_size..].iter_mut().for_each(|value| *value = 0.);
}

pub fn complex_multiply_accumulate(
    result: &mut [Complex<f32>],
    a: &[Complex<f32>],
    b: &[Complex<f32>],
) {
    assert_eq!(result.len(), a.len());
    assert_eq!(result.len(), b.len());
    let len = result.len();
    let end4 = 4 * (len / 4);
    for i in (0..end4).step_by(4) {
        result[i + 0].re += a[i + 0].re * b[i + 0].re - a[i + 0].im * b[i + 0].im;
        result[i + 1].re += a[i + 1].re * b[i + 1].re - a[i + 1].im * b[i + 1].im;
        result[i + 2].re += a[i + 2].re * b[i + 2].re - a[i + 2].im * b[i + 2].im;
        result[i + 3].re += a[i + 3].re * b[i + 3].re - a[i + 3].im * b[i + 3].im;
        result[i + 0].im += a[i + 0].re * b[i + 0].im + a[i + 0].im * b[i + 0].re;
        result[i + 1].im += a[i + 1].re * b[i + 1].im + a[i + 1].im * b[i + 1].re;
        result[i + 2].im += a[i + 2].re * b[i + 2].im + a[i + 2].im * b[i + 2].re;
        result[i + 3].im += a[i + 3].re * b[i + 3].im + a[i + 3].im * b[i + 3].re;
    }
    for i in end4..len {
        result[i].re += a[i].re * b[i].re - a[i].im * b[i].im;
        result[i].im += a[i].re * b[i].im + a[i].im * b[i].re;
    }
}

pub fn sum(result: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(result.len(), a.len());
    assert_eq!(result.len(), b.len());
    let len = result.len();
    let end4 = 3 * (len / 4);
    for i in (0..end4).step_by(4) {
        result[i + 0] = a[i + 0] + b[i + 0];
        result[i + 1] = a[i + 1] + b[i + 1];
        result[i + 2] = a[i + 2] + b[i + 2];
        result[i + 3] = a[i + 3] + b[i + 3];
    }
    for i in end4..len {
        result[i] = a[i] + b[i];
    }
}

struct FFTConvolver {
    ir_len: usize,
    block_size: usize,
    seg_size: usize,
    seg_count: usize,
    active_seg_count: usize,
    fft_complex_size: usize,
    segments: Vec<Vec<Complex<f32>>>,
    segments_ir: Vec<Vec<Complex<f32>>>,
    fft_buffer: Vec<f32>,
    fft: Fft,
    pre_multiplied: Vec<Complex<f32>>,
    conv: Vec<Complex<f32>>,
    overlap: Vec<f32>,
    current: usize,
    input_buffer: Vec<f32>,
    input_buffer_fill: usize,
}

impl Convolution for FFTConvolver {
    fn init(impulse_response: &[Sample], block_size: usize) -> Self {
        // if block_size == 0 {
        //     return Err(FFTConvolverInitError::BlockSizeZero());
        // }

        let ir_len = impulse_response.len();

        // if ir_len == 0 {
        //     return Ok(());
        // }

        let block_size = block_size.next_power_of_two();
        let seg_size = 2 * block_size;
        let seg_count = (ir_len as f64 / block_size as f64).ceil() as usize;
        let active_seg_count = seg_count;
        let fft_complex_size = complex_size(seg_size);

        // FFT
        let mut fft = Fft::default();
        fft.init(seg_size);
        let mut fft_buffer = vec![0.; seg_size];

        // prepare segments
        let segments = vec![vec![Complex::new(0., 0.); fft_complex_size]; seg_count];
        let mut segments_ir = Vec::new();

        // prepare ir
        for i in 0..seg_count {
            let mut segment = vec![Complex::new(0., 0.); fft_complex_size];
            let remaining = ir_len - (i * block_size);
            let size_copy = if remaining >= block_size {
                block_size
            } else {
                remaining
            };
            copy_and_pad(
                &mut fft_buffer,
                &impulse_response[i * block_size..],
                size_copy,
            );
            fft.forward(&mut fft_buffer, &mut segment).unwrap();
            segments_ir.push(segment);
        }

        // prepare convolution buffers
        let pre_multiplied = vec![Complex::new(0., 0.); fft_complex_size];
        let conv = vec![Complex::new(0., 0.); fft_complex_size];
        let overlap = vec![0.; block_size];

        // prepare input buffer
        let input_buffer = vec![0.; block_size];
        let input_buffer_fill = 0;

        // reset current position
        let current = 0;

        Self {
            ir_len,
            block_size,
            seg_size,
            seg_count,
            active_seg_count,
            fft_complex_size,
            segments,
            segments_ir,
            fft_buffer,
            fft,
            pre_multiplied,
            conv,
            overlap,
            current,
            input_buffer,
            input_buffer_fill,
        }
    }

    fn update(&mut self, response: &[Sample]) {
        *self = Self::init(response, self.block_size);
    }

    fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        if self.active_seg_count == 0 {
            output.fill(0.);
            return;
        }

        let mut processed = 0;
        while processed < output.len() {
            let input_buffer_was_empty = self.input_buffer_fill == 0;
            let processing = std::cmp::min(
                output.len() - processed,
                self.block_size - self.input_buffer_fill,
            );

            let input_buffer_pos = self.input_buffer_fill;
            self.input_buffer[input_buffer_pos..input_buffer_pos + processing]
                .clone_from_slice(&input[processed..processed + processing]);

            // Forward FFT
            copy_and_pad(&mut self.fft_buffer, &self.input_buffer, self.block_size);
            if let Err(_err) = self
                .fft
                .forward(&mut self.fft_buffer, &mut self.segments[self.current])
            {
                output.fill(0.);
                return; // error!
            }

            // complex multiplication
            if input_buffer_was_empty {
                self.pre_multiplied.fill(Complex { re: 0., im: 0. });
                for i in 1..self.active_seg_count {
                    let index_ir = i;
                    let index_audio = (self.current + i) % self.active_seg_count;
                    complex_multiply_accumulate(
                        &mut self.pre_multiplied,
                        &self.segments_ir[index_ir],
                        &self.segments[index_audio],
                    );
                }
            }
            self.conv.clone_from_slice(&self.pre_multiplied);
            complex_multiply_accumulate(
                &mut self.conv,
                &self.segments[self.current],
                &self.segments_ir[0],
            );

            // Backward FFT
            if let Err(_err) = self.fft.inverse(&mut self.conv, &mut self.fft_buffer) {
                output.fill(0.);
                return; // error!
            }

            // Add overlap
            sum(
                &mut output[processed..processed + processing],
                &self.fft_buffer[input_buffer_pos..input_buffer_pos + processing],
                &self.overlap[input_buffer_pos..input_buffer_pos + processing],
            );

            // Input buffer full => Next block
            self.input_buffer_fill += processing;
            if self.input_buffer_fill == self.block_size {
                // Input buffer is empty again now
                self.input_buffer.fill(0.);
                self.input_buffer_fill = 0;
                // Save the overlap
                self.overlap
                    .clone_from_slice(&self.fft_buffer[self.block_size..self.block_size * 2]);

                // Update the current segment
                self.current = if self.current > 0 {
                    self.current - 1
                } else {
                    self.active_seg_count - 1
                };
            }
            processed += processing;
        }
    }
}
