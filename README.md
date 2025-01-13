# fft-convolution

A rust implementation of convolution in the frequency domain (suitable for real-time use) inspired by the [HiFi-LoFi FFT Convolver](https://github.com/HiFi-LoFi/FFTConvolver) which is MIT licensed.

As of the time of this writing there does not seem to be a rust implementation of FFT convolution algorithms that is suitable for real-time use. One can find an [older rust port](https://github.com/neodsp/fft-convolver) of just the `FFTConvolver` class, but it does not support real-time switching of impulse responses and does not include implementations for the `TwoStageFFTConvolver` and `CrossfadeConvolver` classes.

As the original C++ implementation, this library implements:

- Partitioned convolution algorithm (using uniform block sizes)
- Optional support for non-uniform block sizes (`TwoStageFFTConvolver`)

On top of that it implements:

- Real-time safe switching of impulse responses in the `FFTConvolver`
- Real-time and artefact-free switching of impulse responses using the `CrossfadeConvolver`

Compared to the original C++ implementation, this implementation does _not_ provide:

- Its own FFT implementation (it currently uses the rustfft crate)
- The option to use SSE instructions in the `FFTConvolver`

## Prerequisites:

- rust >=1.72.0
