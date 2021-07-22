use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array3;

use gaussian_filter::gaussian_filter;

fn bench_local_maxima(c: &mut Criterion) {
    let data = Array3::<f32>::zeros((134, 156, 130));
    c.bench_function("local_maxima", |b| {
        b.iter(|| black_box(gaussian_filter(data.clone(), 1.0, 2.0)));
    });
}

criterion_group!(benches, bench_local_maxima,);
criterion_main!(benches);
