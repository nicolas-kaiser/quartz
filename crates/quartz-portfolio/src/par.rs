//! Parallel-map shim: rayon when the `parallel` feature is enabled (default),
//! plain serial iteration otherwise. Both preserve input order.

#[cfg(feature = "parallel")]
pub(crate) fn par_map<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    use rayon::prelude::*;
    // Indexed par_iter + collect preserves input order.
    items.par_iter().map(f).collect()
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn par_map<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    F: Fn(&T) -> R,
{
    items.iter().map(f).collect()
}
