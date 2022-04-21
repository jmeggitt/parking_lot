// Copyright 2016 Amanieu d'Antras
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

mod args;
use crate::args::ArgRange;

use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};

use std::arch::asm;
#[cfg(any(windows, unix))]
use std::cell::UnsafeCell;
use std::thread::JoinHandle;
use std::time::Instant;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Barrier,
    },
    thread,
    time::Duration,
};

trait WorkLoad<T> {
    fn work(x: &T) -> T;
}

trait Mutex<T> {
    fn new(v: T) -> Self;
    fn lock<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R;
    fn name() -> &'static str;
}

impl<T> Mutex<T> for std::sync::Mutex<T> {
    fn new(v: T) -> Self {
        Self::new(v)
    }
    fn lock<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        f(&mut *self.lock().unwrap())
    }
    fn name() -> &'static str {
        "std::sync::Mutex"
    }
}

impl<T> Mutex<T> for parking_lot::Mutex<T> {
    fn new(v: T) -> Self {
        Self::new(v)
    }
    fn lock<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        f(&mut *self.lock())
    }
    fn name() -> &'static str {
        "parking_lot::Mutex"
    }
}

impl<T> Mutex<T> for parking_lot::FairMutex<T> {
    fn new(v: T) -> Self {
        Self::new(v)
    }
    fn lock<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        f(&mut *self.lock())
    }
    fn name() -> &'static str {
        "parking_lot::FairMutex"
    }
}

/// As a comparison, also test a RwLock that gets used even though no reads occur.
impl<T> Mutex<T> for parking_lot::RwLock<T> {
    fn new(v: T) -> Self {
        Self::new(v)
    }
    fn lock<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        f(&mut *self.write())
    }
    fn name() -> &'static str {
        "parking_lot::RwLock"
    }
}

/// A regular value which pretends to be a mutex. Can be done with regular tests to measure
/// benchmark overhead. Should only be used in single threaded use cases
struct Theoretical<T>(UnsafeCell<T>);

unsafe impl<T> Sync for Theoretical<T> {}
unsafe impl<T> Send for Theoretical<T> {}

impl<T> Mutex<T> for Theoretical<T> {
    fn new(v: T) -> Self {
        Theoretical(UnsafeCell::new(v))
    }

    fn lock<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        unsafe { f(&mut *self.0.get()) }
    }

    fn name() -> &'static str {
        "Overhead"
    }
}

#[cfg(not(windows))]
type SrwLock<T> = std::sync::Mutex<T>;

#[cfg(windows)]
use winapi::um::synchapi;
#[cfg(windows)]
struct SrwLock<T>(UnsafeCell<T>, UnsafeCell<synchapi::SRWLOCK>);
#[cfg(windows)]
unsafe impl<T> Sync for SrwLock<T> {}
#[cfg(windows)]
unsafe impl<T: Send> Send for SrwLock<T> {}
#[cfg(windows)]
impl<T> Mutex<T> for SrwLock<T> {
    fn new(v: T) -> Self {
        let mut h: synchapi::SRWLOCK = synchapi::SRWLOCK {
            Ptr: std::ptr::null_mut(),
        };

        unsafe {
            synchapi::InitializeSRWLock(&mut h);
        }
        SrwLock(UnsafeCell::new(v), UnsafeCell::new(h))
    }
    fn lock<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        unsafe {
            synchapi::AcquireSRWLockExclusive(self.1.get());
            let res = f(&mut *self.0.get());
            synchapi::ReleaseSRWLockExclusive(self.1.get());
            res
        }
    }
    fn name() -> &'static str {
        "winapi_srwlock"
    }
}

#[cfg(not(unix))]
type PthreadMutex<T> = std::sync::Mutex<T>;

#[cfg(unix)]
struct PthreadMutex<T>(UnsafeCell<T>, UnsafeCell<libc::pthread_mutex_t>);
#[cfg(unix)]
unsafe impl<T> Sync for PthreadMutex<T> {}
#[cfg(unix)]
impl<T> Mutex<T> for PthreadMutex<T> {
    fn new(v: T) -> Self {
        PthreadMutex(
            UnsafeCell::new(v),
            UnsafeCell::new(libc::PTHREAD_MUTEX_INITIALIZER),
        )
    }
    fn lock<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        unsafe {
            libc::pthread_mutex_lock(self.1.get());
            let res = f(&mut *self.0.get());
            libc::pthread_mutex_unlock(self.1.get());
            res
        }
    }
    fn name() -> &'static str {
        "pthread_mutex_t"
    }
}
#[cfg(unix)]
impl<T> Drop for PthreadMutex<T> {
    fn drop(&mut self) {
        unsafe {
            libc::pthread_mutex_destroy(self.1.get());
        }
    }
}

fn run_benchmark<M: Mutex<f64> + Send + Sync + 'static>(
    num_threads: usize,
    work_per_critical_section: usize,
    work_between_critical_sections: usize,
    seconds_per_test: usize,
) -> Vec<usize> {
    let lock = Arc::new(([0u8; 300], M::new(0.0), [0u8; 300]));
    let keep_going = Arc::new(AtomicBool::new(true));
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut threads = vec![];
    for _ in 0..num_threads {
        let barrier = barrier.clone();
        let lock = lock.clone();
        let keep_going = keep_going.clone();
        threads.push(thread::spawn(move || {
            let mut local_value = 0.0;
            let mut value = 0.0;
            let mut iterations = 0usize;
            barrier.wait();
            while keep_going.load(Ordering::Relaxed) {
                lock.1.lock(|shared_value| {
                    for _ in 0..work_per_critical_section {
                        *shared_value += value;
                        *shared_value *= 1.01;
                        value = *shared_value;
                    }
                });
                for _ in 0..work_between_critical_sections {
                    local_value += value;
                    local_value *= 1.01;
                    value = local_value;
                }
                iterations += 1;
            }
            (iterations, value)
        }));
    }

    thread::sleep(Duration::from_secs(seconds_per_test as u64));
    keep_going.store(false, Ordering::Relaxed);
    threads.into_iter().map(|x| x.join().unwrap().0).collect()
}

fn run_no_load_iters<M: Mutex<usize> + Send + Sync + 'static>(
    num_threads: usize,
    iters_per_thread: usize,
) -> Duration {
    let lock = Arc::new(M::new(1usize));
    let barrier = Arc::new(Barrier::new(num_threads));

    let mut threads = vec![];
    for i in 1..(num_threads + 1) {
        let barrier = barrier.clone();
        let lock = lock.clone();

        threads.push(thread::spawn(move || {
            barrier.wait();
            let start_time = Instant::now();

            for _ in 0..iters_per_thread {
                lock.lock(|shared_value| {
                    // Surround in black_box to prevent optimization of workload
                    // Perform a single add which mutates the state to prevent shared_value from
                    // being optimized away in the event of black_box failing on some targets
                    *shared_value = black_box(shared_value.wrapping_add(i));
                });
            }

            let ret = start_time.elapsed();
            ret
        }));
    }

    let lock_time = threads
        .into_iter()
        .map(JoinHandle::join)
        .map(Result::unwrap)
        .fold(Duration::from_secs(0), |a, b| a + b);


    // Read the value in the mutex to further discourage optimization
    assert_eq!(lock.lock(|x| *x), 1 + iters_per_thread * num_threads * (num_threads + 1) / 2);

    lock_time
}

fn run_benchmark_iterations<M: Mutex<f64> + Send + Sync + 'static>(
    num_threads: usize,
    work_per_critical_section: usize,
    work_between_critical_sections: usize,
    seconds_per_test: usize,
    test_iterations: usize,
) {
    let mut data = vec![];
    for _ in 0..test_iterations {
        let run_data = run_benchmark::<M>(
            num_threads,
            work_per_critical_section,
            work_between_critical_sections,
            seconds_per_test,
        );
        data.extend_from_slice(&run_data);
    }

    let average = data.iter().fold(0f64, |a, b| a + *b as f64) / data.len() as f64;
    let variance = data
        .iter()
        .fold(0f64, |a, b| a + ((*b as f64 - average).powi(2)))
        / data.len() as f64;
    data.sort();

    let k_hz = 1.0 / seconds_per_test as f64 / 1000.0;
    println!(
        "{:20} | {:10.3} kHz | {:10.3} kHz | {:10.3} kHz",
        M::name(),
        average * k_hz,
        data[data.len() / 2] as f64 * k_hz,
        variance.sqrt() * k_hz
    );
}

fn run_all(
    args: &[ArgRange],
    first: &mut bool,
    num_threads: usize,
    work_per_critical_section: usize,
    work_between_critical_sections: usize,
    seconds_per_test: usize,
    test_iterations: usize,
) {
    if num_threads == 0 {
        return;
    }
    if *first || !args[0].is_single() {
        println!("- Running with {} threads", num_threads);
    }
    if *first || !args[1].is_single() || !args[2].is_single() {
        println!(
            "- {} iterations inside lock, {} iterations outside lock",
            work_per_critical_section, work_between_critical_sections
        );
    }
    if *first || !args[3].is_single() {
        println!("- {} seconds per test", seconds_per_test);
    }
    *first = false;

    println!(
        "{:^20} | {:^14} | {:^14} | {:^14}",
        "name", "average", "median", "std.dev."
    );

    run_benchmark_iterations::<parking_lot::Mutex<f64>>(
        num_threads,
        work_per_critical_section,
        work_between_critical_sections,
        seconds_per_test,
        test_iterations,
    );

    run_benchmark_iterations::<std::sync::Mutex<f64>>(
        num_threads,
        work_per_critical_section,
        work_between_critical_sections,
        seconds_per_test,
        test_iterations,
    );
    if cfg!(windows) {
        run_benchmark_iterations::<SrwLock<f64>>(
            num_threads,
            work_per_critical_section,
            work_between_critical_sections,
            seconds_per_test,
            test_iterations,
        );
    }
    if cfg!(unix) {
        run_benchmark_iterations::<PthreadMutex<f64>>(
            num_threads,
            work_per_critical_section,
            work_between_critical_sections,
            seconds_per_test,
            test_iterations,
        );
    }
}

/// A workload which has little to no effect. Potentially risky option since this relies on
/// `criterion::black_box` working as intended to avoid the compiler optimizations ruining the
/// benchmark. It should work as intended, but we are only given a full guarantee when running on
/// nightly.
struct NoLoad;

impl<T: Copy> WorkLoad<T> for NoLoad {
    fn work(x: &T) -> T {
        *x
    }
}

/// A repeating bit sequence which should be difficult for the compiler to optimize.
/// Tests on x86_64 with rustc 1.60 stable show the best compiler can do is unroll the for loop
/// and get about 200 instructions.
struct PRBS31;

impl WorkLoad<u32> for PRBS31 {
    #[inline(never)]
    fn work(x: &u32) -> u32 {
        let mut value = *x;
        for _ in 0..32 {
            let new_bit = ((value >> 30) ^ (value >> 27)) & 1;
            value = ((value << 1) | new_bit) & ((1u32 << 31) - 1);
        }
        value
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut no_load = c.benchmark_group("no-load");

    // We need to use logarithmic scaling otherwise it may be hard to see some cases
    no_load.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    /// We need to run each action for extra iterations to ensure the mutex is actually has thechance to be contested.
    const ACTIONS_PER_ITER: usize = 1000;

    macro_rules! run_bench {
        ($group:ident, $($mutex:ident)::+, $sub_id:ident, $threads:expr) => {{
            let bench_id = BenchmarkId::new($($mutex)::+::<usize>::name(), $sub_id);

            $group.bench_function(bench_id, |b| {
                b.iter_custom(|i| run_no_load_iters::<$($mutex)::+::<usize>>($threads, i as usize * ACTIONS_PER_ITER) / ACTIONS_PER_ITER as u32)
            });
        }};
    }

    for threads in 1..=16 {
        // Tell criterion we are bumping up the difficulty so it can account for it in the report
        no_load.throughput(Throughput::Elements(threads as u64));

        run_bench!(no_load, std::sync::Mutex, threads, threads);
        run_bench!(no_load, parking_lot::RwLock, threads, threads);
        run_bench!(no_load, parking_lot::FairMutex, threads, threads);
        run_bench!(no_load, parking_lot::Mutex, threads, threads);

        #[cfg(windows)]
        run_bench!(no_load, SrwLock, threads, threads);

        #[cfg(unix)]
        run_bench!(no_load, PthreadMutex, threads, threads);
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
