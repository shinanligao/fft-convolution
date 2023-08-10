use crate::{Conv, EvolveResponse, Sample};

#[derive(Clone)]
pub struct TimeVaryingConvolver {}

impl Conv for TimeVaryingConvolver {
    fn init(_response: &[Sample], _max_block_size: usize) -> Self {
        todo!()
    }

    fn set_response(&mut self, _response: &[Sample]) {
        todo!()
    }

    fn process(&mut self, _input: &[Sample], _output: &mut [Sample]) {
        todo!()
    }
}

impl EvolveResponse for TimeVaryingConvolver {
    fn evolve(&mut self, _response: &[Sample]) {
        self.set_response(_response);
    }
}
