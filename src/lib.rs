pub mod crossfade_convolver;
pub mod fft_convolver;
pub mod time_varying_convolver;

// todo: use a generic floating point type
pub type Sample = f32;

pub trait Conv: Clone {
    fn init(response: &[Sample], max_block_size: usize) -> Self;
    fn set_response(&mut self, response: &[Sample]);
    fn process(&mut self, input: &[Sample], output: &mut [Sample]);
}

pub trait SmoothConvUpdate: Conv {
    fn evolve(&mut self, response: &[Sample]);
}

fn _smooth_convolvers_example() {
    use crate::{
        crossfade_convolver::CrossfadeConvolver, fft_convolver::FFTConvolver,
        time_varying_convolver::TimeVaryingConvolver,
    };
    struct AudioNode<Convolver: SmoothConvUpdate> {
        _convolver: Convolver,
    }

    let _node = AudioNode {
        _convolver: TimeVaryingConvolver::init(&[0.0; 1024], 1024),
    };

    let _node2 = AudioNode {
        _convolver: CrossfadeConvolver::new(FFTConvolver::init(&[0.0; 1024], 1024), 2048, 512, 256),
    };
}
