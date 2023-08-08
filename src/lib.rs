use rustfft::{num_complex::Complex, Fft};
use std::sync::Arc;

// todo: use a generic floating point type
type Sample = f32;

pub struct Response {
    data: Vec<Complex<Sample>>,
}

impl Response {
    pub fn new(length: usize) -> Self {
        Self {
            data: vec![Complex::new(1.0, 0.0); length],
        }
    }
}

pub trait Convolve {
    fn init(response: &[Sample], max_block_size: usize) -> Self;
    fn set_response(&mut self, response: &[Sample]);
    fn convolve(&mut self, input: &[Sample], output: &mut [Sample]);
}

struct FFTConvolver;
impl Convolve for FFTConvolver {
    fn init(response: &[Sample], max_block_size: usize) -> Self {
        Self
    }

    fn set_response(&mut self, response: &[Sample]) {}

    fn convolve(&mut self, input: &[Sample], output: &mut [Sample]) {
        assert!(input.len() == output.len());
        output.copy_from_slice(input);
    }
}

pub struct Convolver<Processor: Convolve> {
    processor: Processor,
}

impl<Processor: Convolve> Convolver<Processor> {
    pub fn new(convolver_processor: Processor) -> Self {
        Self {
            processor: convolver_processor,
        }
    }

    pub fn set_response(&mut self, response: &[Sample]) {
        self.processor.set_response(response);
    }
    pub fn convolve(&mut self, input: &[Sample], output: &mut [Sample]) {
        self.processor.convolve(input, output);
    }
}

#[test]
fn test_fft_convolver() {
    let mut convolver = Convolver::new(FFTConvolver::init(&[1.0; 1024], 1024));
    let mut input = vec![1.0; 1024];
    let mut output = vec![0.0; 1024];
    convolver.convolve(&input, &mut output);
    assert_eq!(output, vec![1.0; 1024]);
}
