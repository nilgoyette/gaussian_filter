//! This modules defines some image filters.

use ndarray::{Array, Axis, Dimension, ScalarOperand};
use num_traits::{Float, ToPrimitive};

/// Gaussian filter for n-dimensional arrays.
///
/// * `data` - The input N-D data.
/// * `sigma` - Standard deviation for Gaussian kernel.
/// * `truncate` - Truncate the filter at this many standard deviations.
///
/// See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
pub fn gaussian_filter<A, D>(mut data: Array<A, D>, sigma: A, truncate: A) -> Array<A, D>
where
    A: std::ops::AddAssign + Float + ScalarOperand + ToPrimitive + std::fmt::Display,
    D: Dimension,
{
    // Make the radius of the filter equal to truncate standard deviations
    let lw = (truncate * sigma + A::from(0.5).unwrap()).to_usize().unwrap();
    let weights = weights(sigma, lw);
    let half = weights.len() / 2;
    let middle_weight = weights[half];

    for d in 0..data.ndim() {
        let n = data.len_of(Axis(d));
        if half >= n {
            panic!("Data size is too small for the inputs (sigma and truncate)");
        }
        assert!(half < n);

        let mut output = Array::zeros(data.dim());
        let mut buffer = vec![A::zero(); n + 2 * half];
        let input_it = data.lanes(Axis(d)).into_iter();
        let output_it = output.lanes_mut(Axis(d)).into_iter();
        for (input, mut o) in input_it.zip(output_it) {
            unsafe {
                // Prepare the 'reflect' buffer
                let mut pos_b = 0;
                let mut pos_i = half - 1;
                for _ in 0..half {
                    *buffer.get_unchecked_mut(pos_b) = *input.uget(pos_i);
                    pos_b += 1;
                    pos_i = pos_i.wrapping_sub(1);
                }
                let mut pos_i = 0;
                for _ in 0..n {
                    *buffer.get_unchecked_mut(pos_b) = *input.uget(pos_i);
                    pos_b += 1;
                    pos_i += 1;
                }
                pos_i = n - 1;
                for _ in 0..half {
                    *buffer.get_unchecked_mut(pos_b) = *input.uget(pos_i);
                    pos_b += 1;
                    pos_i = pos_i.wrapping_sub(1);
                }

                // Convolve the input data with the weights array.
                for idx in 0..n {
                    let s = half + idx;
                    let mut pos_l = s - 1;
                    let mut pos_r = s + 1;

                    let mut sum = *buffer.get_unchecked(s) * middle_weight;
                    for &w in &weights[half + 1..] {
                        sum += (*buffer.get_unchecked(pos_l) + *buffer.get_unchecked(pos_r)) * w;
                        pos_l = pos_l.wrapping_sub(1);
                        pos_r += 1;
                    }
                    *o.uget_mut(idx) = sum;
                }
            }
        }
        data = output;
    }
    data
}

/// Computes a 1-D Gaussian convolution kernel.
fn weights<A>(sigma: A, radius: usize) -> Vec<A>
where
    A: Float + ScalarOperand,
{
    let sigma2 = sigma.powi(2);
    let radius = radius as isize;
    let mut phi_x: Vec<_> = (-radius..=radius)
        .map(|x| (A::from(-0.5).unwrap() / sigma2 * A::from(x.pow(2)).unwrap()).exp())
        .collect();
    let sum = phi_x.iter().fold(A::zero(), |acc, &v| acc + v);
    phi_x.iter_mut().for_each(|v| *v = *v / sum);
    phi_x
}
