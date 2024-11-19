use crate::Sample;

pub trait Mixer {
    fn mix(&self, a: Sample, b: Sample, value: Sample) -> Sample;
}

#[allow(dead_code)]
pub struct LinearMixer;
impl Mixer for LinearMixer {
    fn mix(&self, a: Sample, b: Sample, value: Sample) -> Sample {
        a * (1.0 - value) + b * value
    }
}

#[allow(dead_code)]
pub struct SquareRootMixer;
impl Mixer for SquareRootMixer {
    fn mix(&self, a: Sample, b: Sample, value: Sample) -> Sample {
        let gain1 = (1.0 - value).sqrt();
        let gain2 = value.sqrt();
        a * gain1 + b * gain2
    }
}
const PI_HALF: Sample = std::f32::consts::PI * 0.5;

#[allow(dead_code)]
pub struct CosineMixer;
impl Mixer for CosineMixer {
    fn mix(&self, a: Sample, b: Sample, value: Sample) -> Sample {
        let rad = PI_HALF * value;
        let gain1 = rad.cos();
        let gain2 = rad.sin();
        a * gain1 + b * gain2
    }
}

#[derive(Clone)]
pub struct RaisedCosineMixer;
impl Mixer for RaisedCosineMixer {
    fn mix(&self, a: Sample, b: Sample, value: Sample) -> Sample {
        let rad = PI_HALF * value;
        let gain1 = rad.cos().powi(2);
        let gain2 = 1.0 - gain1;
        a * gain1 + b * gain2
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Target {
    A,
    B,
}

#[derive(Clone, Copy, PartialEq)]
pub enum FadingState {
    Reached(Target),
    Approaching(Target),
}

impl FadingState {
    pub fn target(&self) -> Target {
        match self {
            Self::Reached(target) => *target,
            Self::Approaching(target) => *target,
        }
    }
}

#[derive(Clone)]
pub struct Crossfader<T: Mixer> {
    mixer: T,
    fading_samples: i64,
    hold_samples: i64,
    counter: i64,
    mix_value_step: Sample,
    mix_value: Sample,
    fading_state: FadingState,
}

impl<T: Mixer> Crossfader<T> {
    pub fn new(mixer: T, fading_samples: usize, hold_samples: usize) -> Self {
        Self {
            mixer,
            fading_samples: fading_samples as i64,
            hold_samples: hold_samples as i64,
            counter: 0,
            mix_value_step: 1.0 / fading_samples as Sample,
            mix_value: 0.0,
            fading_state: FadingState::Reached(Target::A),
        }
    }

    pub fn fading_state(&self) -> FadingState {
        self.fading_state
    }

    pub fn fade_into(&mut self, target: Target) {
        let current_target = self.fading_state.target();
        if current_target == target {
            return;
        }

        match self.fading_state {
            FadingState::Reached(_) => {
                self.counter = -self.hold_samples;
                self.fading_state = FadingState::Approaching(target);
                self.mix_value_step = -self.mix_value_step;
            }
            FadingState::Approaching(_) => {
                // note: should never be the case in the context of the crossfade convolver,
                // which will swap responses only after a target is reached
                if self.counter >= 0 {
                    self.counter = self.fading_samples - self.counter;
                    self.fading_state = FadingState::Approaching(target);
                    self.mix_value_step = -self.mix_value_step;
                } else {
                    self.fading_state = FadingState::Reached(target);
                }
            }
        }
    }

    pub fn mix(&mut self, a: Sample, b: Sample) -> Sample {
        match self.fading_state {
            FadingState::Reached(target) => match target {
                Target::A => return a,
                Target::B => return b,
            },
            FadingState::Approaching(target) => {
                self.counter += 1;

                if self.counter <= 0 {
                    // holding the previous target
                    match target {
                        Target::A => return b,
                        Target::B => return a,
                    }
                }

                self.mix_value += self.mix_value_step;

                if self.counter == self.fading_samples {
                    self.fading_state = FadingState::Reached(target);
                    match target {
                        Target::A => {
                            self.mix_value = 0.0;
                            return a;
                        }
                        Target::B => {
                            self.mix_value = 1.0;
                            return b;
                        }
                    }
                }

                self.mixer.mix(a, b, self.mix_value)
            }
        }
    }
}

#[test]
fn test_crossfader() {
    let hold_samples = 4;
    let fading_samples = 4;
    let sample_a = 1.0;
    let sample_b = 10.0;
    let mut crossfader =
        Crossfader::<RaisedCosineMixer>::new(RaisedCosineMixer, fading_samples, hold_samples);

    let start = |target: Target| match target {
        Target::A => sample_b,
        Target::B => sample_a,
    };
    let end = |target: Target| match target {
        Target::A => sample_a,
        Target::B => sample_b,
    };

    for target in [Target::B, Target::A] {
        crossfader.fade_into(target);
        for i in 0..hold_samples + fading_samples {
            let mixed_value = crossfader.mix(sample_a, sample_b);
            if i < hold_samples {
                assert!(crossfader.fading_state == FadingState::Approaching(target));
                assert_eq!(mixed_value, start(target));
            } else if i < hold_samples + fading_samples - 1 {
                assert!(crossfader.fading_state == FadingState::Approaching(target));
                assert_ne!(mixed_value, start(target));
                assert_ne!(mixed_value, end(target));
            } else {
                assert_eq!(mixed_value, end(target));
                assert!(crossfader.fading_state == FadingState::Reached(target));
            }
        }
    }
}
