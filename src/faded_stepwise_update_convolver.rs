use crate::fft_convolver::{complex_size, copy_and_pad, FFTConvolverOLA};
use crate::{Convolution, Sample};
use rustfft::num_complex::Complex;

#[derive(Clone)]
struct ImpulseResponse {
    samples: Vec<Sample>,
    effective_length: usize,
}

impl ImpulseResponse {
    pub fn new(max_response_length: usize) -> Self {
        Self {
            samples: vec![0.0; max_response_length],
            effective_length: 0,
        }
    }

    pub fn init(response: &[Sample], max_response_length: usize) -> Self {
        let mut ir = Self::new(max_response_length);
        ir.update(response);
        ir
    }

    pub fn update(&mut self, response: &[Sample]) {
        let response_len = response.len();
        assert!(response_len <= self.samples.len());
        copy_and_pad(&mut self.samples[..], response, response_len);
        self.effective_length = response_len;
    }
}

#[derive(Clone)]
pub struct FadedStepwiseUpdateConvolver {
    convolver: FFTConvolverOLA,
    block_size: usize,
    current_response: ImpulseResponse,
    next_response: ImpulseResponse,
    current_segments: Vec<Vec<Complex<Sample>>>,
    next_segments: Vec<Vec<Complex<Sample>>>,
    queued_response: ImpulseResponse,
    transition_samples: usize,
    fade_steps: usize,
    transition_counter: usize,
    switching: bool,
    response_pending: bool,
}

impl FadedStepwiseUpdateConvolver {
    pub fn new(
        response: &[Sample],
        max_response_length: usize,
        block_size: usize,
        transition_samples: usize,
    ) -> Self {
        if transition_samples < max_response_length {
            println!("The transition cannot be shorter than the max response length.");
        }

        let convolver = FFTConvolverOLA::init(response, block_size, max_response_length);

        Self {
            convolver: convolver.clone(),
            block_size,
            current_response: ImpulseResponse::init(response, max_response_length),
            next_response: ImpulseResponse::new(max_response_length),
            current_segments: convolver.segments_ir().to_vec(),
            next_segments: vec![
                vec![Complex::new(0.0, 0.0); complex_size(2 * block_size)];
                *convolver.active_seg_count()
            ],
            queued_response: ImpulseResponse::new(max_response_length),
            transition_samples: transition_samples.max(max_response_length),
            fade_steps: 0,
            transition_counter: 0,
            switching: false,
            response_pending: false,
        }
    }

    fn initiate_switch(&mut self) {
        self.switching = true;
        self.response_pending = false;
        self.fade_steps = (self.transition_samples
            - self
                .current_response
                .effective_length
                .max(self.next_response.effective_length))
            / self.block_size;
    }
}

impl Convolution for FadedStepwiseUpdateConvolver {
    fn init(response: &[Sample], max_block_size: usize, max_response_length: usize) -> Self {
        Self::new(
            response,
            max_response_length,
            max_block_size,
            max_response_length,
        )
    }

    fn update(&mut self, response: &[Sample]) {
        if !self.switching {
            self.next_response.update(response);
            self.initiate_switch();
            return;
        }

        self.queued_response.update(response);
        self.response_pending = true;
    }

    fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        let mut processed = 0;
        while processed < output.len() {
            let processing = std::cmp::min(
                output.len() - processed,
                self.block_size - *self.convolver.input_buffer_fill(),
            );

            if *self.convolver.input_buffer_fill() == 0 {
                if !self.switching && self.response_pending {
                    std::mem::swap(&mut self.next_response, &mut self.queued_response);
                    self.initiate_switch();
                }

                if self.switching {
                    let mut weight = 0.0;
                    if self.transition_counter < *self.convolver.active_seg_count() {
                        self.convolver.transform_segment(
                            &self.next_response.samples,
                            self.transition_counter,
                            &mut self.next_segments[self.transition_counter],
                        );
                    }
                    for i in 0..*self.convolver.active_seg_count() {
                        let transition_index = self.transition_counter as i64 + 1 - i as i64;
                        weight = if self.fade_steps == 0 {
                            1.0
                        } else {
                            (transition_index as f32 / self.fade_steps as f32)
                                .min(1.0)
                                .max(0.0)
                        };
                        self.convolver.mix_to_segment(
                            &self.current_segments,
                            &self.next_segments,
                            weight,
                            i,
                        );
                    }
                    self.transition_counter += 1;
                    if weight == 1.0 {
                        // last weight is 1.0
                        std::mem::swap(&mut self.current_response, &mut self.next_response);
                        std::mem::swap(&mut self.current_segments, &mut self.next_segments);
                        self.switching = false;
                        self.transition_counter = 0;
                    }
                }
            }

            self.convolver.process(
                &input[processed..processed + processing],
                &mut output[processed..processed + processing],
            );

            processed += processing;
        }
    }
}

#[test]
fn test_crossfade_convolver_passthrough() {
    let mut response = [0.0; 1024];
    response[0] = 1.0;
    let mut convolver = FadedStepwiseUpdateConvolver::new(&response, 1024, 1024, response.len());
    let input = vec![1.0; 1024];
    let mut output = vec![0.0; 1024];
    convolver.process(&input, &mut output);

    for i in 0..1024 {
        assert!((output[i] - 1.0).abs() < 1e-6);
    }
}
