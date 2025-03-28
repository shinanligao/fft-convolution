use crate::crossfader::{FadingState, Target};
use crate::fft_convolver::{complex_multiply_accumulate, complex_size, copy_and_pad, Fft};
use crate::{Convolution, Sample};
use rustfft::num_complex::Complex;

#[derive(Clone)]
pub struct CrossfadeConvolverFrequencyDomainCore {
    ir_len: usize,
    block_size: usize,
    seg_count: usize,
    segments_ir: Vec<Vec<Complex<Sample>>>,
    fft_buffer: Vec<Sample>,
    fft: Fft,
    conv: Vec<Complex<Sample>>,
}

impl CrossfadeConvolverFrequencyDomainCore {
    fn update(&mut self, response: &[Sample], active_seg_count: usize) {
        let new_ir_len = response.len();

        if new_ir_len > self.ir_len {
            panic!("New impulse response is longer than initialized length");
        }

        if new_ir_len == 0 {
            return;
        }

        self.fft_buffer.fill(0.);
        self.conv.fill(Complex::new(0., 0.));

        // self.active_seg_count = ((new_ir_len as f64 / self.block_size as f64).ceil()) as usize;

        // Prepare IR
        for i in 0..active_seg_count {
            let segment = &mut self.segments_ir[i];
            let remaining = new_ir_len - (i * self.block_size);
            let size_copy = if remaining >= self.block_size {
                self.block_size
            } else {
                remaining
            };
            copy_and_pad(
                &mut self.fft_buffer,
                &response[i * self.block_size..],
                size_copy,
            );
            self.fft.forward(&mut self.fft_buffer, segment).unwrap();
        }

        // Clear remaining segments
        for i in active_seg_count..self.seg_count {
            self.segments_ir[i].fill(Complex::new(0., 0.));
        }
    }
}

#[derive(Clone)]
pub struct CrossfadeConvolverFrequencyDomain {
    block_size: usize,
    active_seg_count: usize,
    segments: Vec<Vec<Complex<Sample>>>,
    core_a: CrossfadeConvolverFrequencyDomainCore,
    core_b: CrossfadeConvolverFrequencyDomainCore,
    conv: Vec<Complex<Sample>>,
    fft_buffer: Vec<Sample>,
    input_buffer: Vec<Sample>,
    current: usize,
    stored_response: Vec<Sample>,
    response_pending: bool,
    fading_state: FadingState,
}

impl CrossfadeConvolverFrequencyDomain {
    pub fn is_crossfading(&self) -> bool {
        self.response_pending
    }

    fn swap(&mut self, response: &[Sample]) {
        // hacky
        self.active_seg_count = std::cmp::max(
            self.active_seg_count,
            ((response.len() as f64 / self.block_size as f64).ceil()) as usize,
        );
        match self.fading_state.target() {
            Target::A => {
                self.core_b.update(response, self.active_seg_count);
                self.fade_into(Target::B);
            }
            Target::B => {
                self.core_a.update(response, self.active_seg_count);
                self.fade_into(Target::A);
            }
        }
    }

    fn fade_into(&mut self, target: Target) {
        let current_target = self.fading_state.target();
        if current_target == target {
            return;
        }

        match self.fading_state {
            FadingState::Reached(_) => {
                self.fading_state = FadingState::Approaching(target);
            }
            FadingState::Approaching(_) => {
                // do nothing, we shouldn't get here
                println!("OOPS");
            }
        }
    }
}

impl Convolution for CrossfadeConvolverFrequencyDomain {
    fn init(impulse_response: &[Sample], block_size: usize, max_response_length: usize) -> Self {
        if max_response_length < impulse_response.len() {
            panic!(
                "max_response_length must be at least the length of the initial impulse response"
            );
        }
        let stored_response = vec![0.; max_response_length];
        let mut padded_ir = impulse_response.to_vec();
        padded_ir.resize(max_response_length, 0.);
        let ir_len = padded_ir.len();

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
            copy_and_pad(&mut fft_buffer, &padded_ir[i * block_size..], size_copy);
            fft.forward(&mut fft_buffer, &mut segment).unwrap();
            segments_ir.push(segment);
        }

        // prepare convolution buffers
        let conv = vec![Complex::new(0., 0.); fft_complex_size];

        let core = CrossfadeConvolverFrequencyDomainCore {
            ir_len,
            block_size,
            seg_count,
            segments_ir,
            fft_buffer: fft_buffer.clone(),
            fft,
            conv: conv.clone(),
        };

        let input_buffer = vec![0.; seg_size];

        // reset current position
        let current = 0;

        Self {
            block_size,
            active_seg_count,
            segments,
            core_a: core.clone(),
            core_b: core.clone(),
            conv,
            input_buffer,
            fft_buffer,
            current,
            stored_response,
            response_pending: false,
            fading_state: FadingState::Reached(Target::A),
        }
    }

    fn update(&mut self, response: &[Sample]) {
        if !self.is_crossfading() {
            self.swap(response);
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
        if self.active_seg_count == 0 {
            output.fill(0.);
            return;
        }

        self.input_buffer.copy_within(self.block_size.., 0);
        self.input_buffer[self.block_size..2 * self.block_size].clone_from_slice(&input);

        self.fft_buffer.copy_from_slice(&self.input_buffer);

        // Forward FFT
        if let Err(_err) = self
            .core_a
            .fft
            .forward(&mut self.fft_buffer, &mut self.segments[self.current])
        {
            output.fill(0.);
            return; // error!
        }

        self.core_a.conv.fill(Complex { re: 0., im: 0. });
        for i in 0..self.active_seg_count {
            let index_ir = i;
            let index_audio = (self.current + i) % self.active_seg_count;
            complex_multiply_accumulate(
                &mut self.core_a.conv,
                &self.core_a.segments_ir[index_ir],
                &self.segments[index_audio],
            );
        }

        self.core_b.conv.fill(Complex { re: 0., im: 0. });
        for i in 0..self.active_seg_count {
            let index_ir = i;
            let index_audio = (self.current + i) % self.active_seg_count;
            complex_multiply_accumulate(
                &mut self.core_b.conv,
                &self.core_b.segments_ir[index_ir],
                &self.segments[index_audio],
            );
        }

        match self.fading_state {
            FadingState::Reached(target) => {
                // do nothing
                match target {
                    Target::A => {
                        self.conv.copy_from_slice(&self.core_a.conv);
                    }
                    Target::B => {
                        self.conv.copy_from_slice(&self.core_b.conv);
                    }
                }
            }
            FadingState::Approaching(target) => match target {
                Target::A => {
                    apply_fading_envelopes(
                        complex_size(2 * self.block_size),
                        &self.core_b.conv,
                        &self.core_a.conv,
                        &mut self.conv,
                    );
                    self.fading_state = FadingState::Reached(Target::A);
                }
                Target::B => {
                    apply_fading_envelopes(
                        complex_size(2 * self.block_size),
                        &self.core_a.conv,
                        &self.core_b.conv,
                        &mut self.conv,
                    );
                    self.fading_state = FadingState::Reached(Target::B);
                }
            },
        }

        // Backward FFT
        if let Err(_err) = self
            .core_a
            .fft
            .inverse(&mut self.conv, &mut self.fft_buffer)
        {
            output.fill(0.);
            println!("Error: {}", _err);
            return; // error!
        }

        output.copy_from_slice(&self.fft_buffer[self.block_size..2 * self.block_size]);

        // Update the current segment
        self.current = if self.current > 0 {
            self.current - 1
        } else {
            self.active_seg_count - 1
        };
    }
}

fn apply_fading_envelopes(
    complex_size: usize,
    fade_out: &[Complex<Sample>],
    fade_in: &[Complex<Sample>],
    result: &mut [Complex<Sample>],
) {
    let get_circular_complex = |i: i32, values: &[Complex<Sample>]| {
        let complex_size = complex_size;
        if i < 0 {
            values[i.abs() as usize].conj()
        } else if i >= complex_size as i32 {
            values[2 * complex_size - i as usize - 2].conj()
        } else {
            values[i as usize]
        }
    };
    for i in 0..complex_size {
        result[i] = 0.5
            * (fade_out[i]
                + fade_in[i]
                + 0.5
                    * (get_circular_complex(i as i32 + 1, fade_in)
                        - get_circular_complex(i as i32 + 1, fade_out)
                        + get_circular_complex(i as i32 - 1, fade_in)
                        - get_circular_complex(i as i32 - 1, fade_out)));
        // hacky, to avoid an error during inverse FFT
        if let (true, im) = ((i == 0 || i == complex_size - 1), result[i].im) {
            result[i].im = if im.abs() < f32::EPSILON { 0.0 } else { im };
        }
    }
}

#[test]
fn test_crossfade_convolver_fd_passthrough() {
    let mut response = [0.0; 1024];
    response[0] = 1.0;
    let mut convolver = CrossfadeConvolverFrequencyDomain::init(&response, 1024, 1024);
    let input = vec![1.0; 1024];
    let mut output = vec![0.0; 1024];
    convolver.process(&input, &mut output);

    for i in 0..1024 {
        assert!((output[i] - 1.0).abs() < 1e-6);
    }
}
