use crate::{Conv, EvolveResponse, Sample};

#[derive(Clone)]
pub struct CrossfadeConvolver<Convolver> {
    convolver_a: Convolver,
    convolver_b: Convolver,
    _crossfade_samples: usize,
    _crossfade_counter: usize,
}

impl<T: Conv> CrossfadeConvolver<T> {
    pub fn new(convolver: T, crossfade_samples: usize) -> Self {
        Self {
            convolver_a: convolver.clone(),
            convolver_b: convolver,
            _crossfade_samples: crossfade_samples,
            _crossfade_counter: 0,
        }
    }
}

impl<Convolver: Conv> Conv for CrossfadeConvolver<Convolver> {
    fn init(response: &[Sample], max_block_size: usize) -> Self {
        let processor = Convolver::init(response, max_block_size);
        Self::new(processor, 0)
    }

    fn set_response(&mut self, response: &[Sample]) {
        self.convolver_a.set_response(response);
        self.convolver_b.set_response(response);
    }

    fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        self.convolver_a.process(input, output);
    }
}

impl<Convolver: Conv> EvolveResponse for CrossfadeConvolver<Convolver> {
    fn evolve(&mut self, response: &[Sample]) {
        // TODO: crossfade and swap...
        self.convolver_a.set_response(response);
        self.convolver_b.set_response(response);
    }
}

#[test]
fn test_crossfade_convolver() {
    let mut response = [0.0; 1024];
    response[0] = 1.0;
    let mut convolver = CrossfadeConvolver::new(
        crate::fft_convolver::FFTConvolver::init(&response, 1024),
        1024,
    );
    let input = vec![1.0; 1024];
    let mut output = vec![0.0; 1024];
    convolver.process(&input, &mut output);

    for i in 0..1024 {
        assert!((output[i] - 1.0).abs() < 1e-6);
    }
}
