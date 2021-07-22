//! This modules defines some image filters.

use ndarray::{s, Array, Array1, ArrayView1, Axis, Dimension, ScalarOperand, Zip};
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
    A: std::ops::AddAssign + Float + ScalarOperand + ToPrimitive,
    D: Dimension,
{
    // Make the radius of the filter equal to truncate standard deviations
    let lw = (truncate * sigma + A::from(0.5).unwrap()).to_usize().unwrap();
    let weights = weights(sigma, lw);
    let nb_weights = weights.len();
    let half = nb_weights / 2;

    for d in 0..data.ndim() {
        let n = data.len_of(Axis(d));
        if half >= n {
            panic!("Data size is too small for the inputs (sigma and truncate)");
        }

        let mut output = Array::zeros(data.dim());
        let mut buffer = Array1::zeros(n + 2 * half);
        Zip::from(data.lanes(Axis(d))).and(output.lanes_mut(Axis(d))).for_each(|input, mut o| {
            fill_buffer(input, &mut buffer, n, half);

            // Convolve the input data with the weights array.
            for idx in 0..n {
                o[idx] = buffer.slice(s![idx..idx + nb_weights]).dot(&weights);
            }
        });
        data = output;
    }
    data
}

/// Computes a 1-D Gaussian convolution kernel.
fn weights<A>(sigma: A, radius: usize) -> Array1<A>
where
    A: Float + ScalarOperand,
{
    let sigma2 = sigma.powi(2);
    let radius = radius as isize;
    let phi_x: Array1<_> = (-radius..=radius)
        .map(|x| (A::from(-0.5).unwrap() / sigma2 * A::from(x.pow(2)).unwrap()).exp())
        .collect();
    let sum = phi_x.iter().fold(A::zero(), |acc, &v| acc + v);
    phi_x / sum
}

/// Prepare the 'reflect' buffer.
fn fill_buffer<A>(input: ArrayView1<A>, buffer: &mut Array1<A>, n: usize, half: usize)
where
    A: Clone + Copy,
{
    for i in 0..half {
        buffer[i] = input[half - i - 1];
    }
    buffer.slice_mut(s![half..half + n]).assign(&input);
    for i in 0..half {
        buffer[n + half + i] = input[n - i - 1];
    }
}
