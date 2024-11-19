use crate::crossfader::{Crossfader, FadingState, RaisedCosineMixer, Target};
use crate::{Convolution, Sample};

#[derive(Clone)]
struct CrossfadeConvolverTimeDomainCore<T: Convolution> {
    convolver_a: T,
    convolver_b: T,
    crossfader: Crossfader<RaisedCosineMixer>,
}

#[derive(Clone)]
pub struct CrossfadeConvolverTimeDomain<Convolver: Convolution> {
    core: CrossfadeConvolverTimeDomainCore<Convolver>,
    buffer_a: Vec<Sample>,
    buffer_b: Vec<Sample>,
    stored_response: Vec<Sample>,
    response_pending: bool,
}

impl<T: Convolution> CrossfadeConvolverTimeDomain<T> {
    pub fn new(
        convolver: T,
        max_response_length: usize,
        max_buffer_size: usize,
        crossfade_samples: usize,
    ) -> Self {
        let stored_response = vec![0.0; max_response_length];
        Self {
            core: CrossfadeConvolverTimeDomainCore {
                convolver_a: convolver.clone(),
                convolver_b: convolver,
                crossfader: Crossfader::new(
                    RaisedCosineMixer,
                    crossfade_samples,
                    max_buffer_size.min(max_response_length),
                ),
            },
            buffer_a: vec![0.0; max_buffer_size],
            buffer_b: vec![0.0; max_buffer_size],
            stored_response,
            response_pending: false,
        }
    }
}

impl<Convolver: Convolution> Convolution for CrossfadeConvolverTimeDomain<Convolver> {
    fn init(response: &[Sample], max_block_size: usize, max_response_length: usize) -> Self {
        let convolver = Convolver::init(response, max_block_size, max_response_length);
        Self::new(convolver, response.len(), max_block_size, response.len())
    }

    fn update(&mut self, response: &[Sample]) {
        if !self.is_crossfading() {
            swap(&mut self.core, response);
            self.response_pending = false;
            return;
        }

        let response_len = response.len();
        assert!(response_len <= self.stored_response.len());

        self.stored_response[..response_len].copy_from_slice(response);
        self.stored_response[response_len..].fill(0.0);
        self.response_pending = true;
    }

    fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        if !self.is_crossfading() && self.response_pending {
            swap(&mut self.core, &mut self.stored_response);
            self.response_pending = false;
        }

        self.core.convolver_a.process(input, &mut self.buffer_a);
        self.core.convolver_b.process(input, &mut self.buffer_b);

        for i in 0..output.len() {
            output[i] = self.core.crossfader.mix(self.buffer_a[i], self.buffer_b[i]);
        }
    }
}

impl<Convolver: Convolution> CrossfadeConvolverTimeDomain<Convolver> {
    pub fn is_crossfading(&self) -> bool {
        match self.core.crossfader.fading_state() {
            FadingState::Approaching(_) => true,
            FadingState::Reached(_) => false,
        }
    }
}

fn swap<T: Convolution>(core: &mut CrossfadeConvolverTimeDomainCore<T>, response: &[Sample]) {
    match core.crossfader.fading_state().target() {
        Target::A => {
            core.convolver_b.update(response);
            core.crossfader.fade_into(Target::B);
        }
        Target::B => {
            core.convolver_a.update(response);
            core.crossfader.fade_into(Target::A);
        }
    }
}

#[test]
fn test_crossfade_convolver_passthrough() {
    let mut response = [0.0; 1024];
    response[0] = 1.0;
    let mut convolver = CrossfadeConvolverTimeDomain::new(
        crate::fft_convolver::FFTConvolverOLA::init(&response, 1024, response.len()),
        1024,
        1024,
        1024,
    );
    let input = vec![1.0; 1024];
    let mut output = vec![0.0; 1024];
    convolver.process(&input, &mut output);

    for i in 0..1024 {
        assert!((output[i] - 1.0).abs() < 1e-6);
    }
}
