// Copyright 2016 Amanieu d'Antras
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};

use criterion::measurement::Measurement;
use std::cell::RefCell;
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

mod mutex_types;

use mutex_types::*;



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

fn bench_threads<P, F, A>(thread_count: usize, mut prepare: P, execute: F) -> Duration
where
    P: FnMut() -> A,
    F: Fn(usize, A) -> Duration + Copy + Send + 'static,
    A: Send + 'static,
{
    let mut threads = vec![];
    let start_barrier = Arc::new(Barrier::new(thread_count));
    let config_barrier = Arc::new(Barrier::new(thread_count));

    for thread_idx in 0..thread_count {
        let start_barrier = start_barrier.clone();
        let config_barrier = config_barrier.clone();
        let args = prepare();
        threads.push(thread::spawn(move || {
            // Wait until all threads are started before attempting to set priority.
            start_barrier.wait();

            #[cfg(windows)]
            unsafe {
                use libc::c_int;
                use winapi::um::processthreadsapi::{GetCurrentThread, SetThreadPriority};
                use winapi::um::winbase::{THREAD_PRIORITY_HIGHEST, THREAD_PRIORITY_TIME_CRITICAL};

                let native_thread = GetCurrentThread();
                let priority;

                // If the requested thread count is less than the available number of cores and the
                // correct features are enabled, elevate to a critical priority level.
                if thread_count <= num_cpus::get() && cfg!(features = "critical-bench") {
                    priority = THREAD_PRIORITY_TIME_CRITICAL;
                } else {
                    priority = THREAD_PRIORITY_HIGHEST;
                }

                if SetThreadPriority(native_thread, priority as c_int) == 0 {
                    panic!("Failed to set thread priority")
                }
            }

            #[cfg(unix)]
            unsafe {
                use libc::{
                    pthread_self, pthread_setschedparam, sched_get_priority_max, sched_param,
                    SCHED_FIFO,
                };

                let params = sched_param {
                    sched_priority: sched_get_priority_max(SCHED_FIFO),
                };

                if params.sched_priority == -1 {
                    panic!("Unable to get max priority");
                }

                let native_thread = pthread_self();
                pthread_setschedparam(native_thread, SCHED_FIFO, &params as *const sched_param);
            }

            // Wait for all threads to be configured then start the test
            config_barrier.wait();
            execute(thread_idx + 1, args)
        }));
    }

    threads
        .into_iter()
        .map(JoinHandle::join)
        .map(Result::unwrap)
        .fold(Duration::from_secs(0), |a, b| a + b)
}

/// Similar to `run_no_load_iters`, but with a completely empty workload. This run makes no
/// attempt to trick the compiler into optimizing away the lock. If the compiler is able to
/// optimize away the lock then it may show that a given mutex is better at taking advantage of
/// compiler optimizations and may yield more favorable machine code. On the other hand if the
/// compiler is unable to optimize it away it still gives up a good base case comparison between the
/// different mutexes.
#[inline(never)]
fn run_true_no_load_iters<M: Mutex<usize> + Send + Sync + 'static>(
    num_threads: usize,
    iters_per_thread: usize,
) -> Duration {
    let lock = Arc::new(M::new(0usize));

    bench_threads(
        num_threads,
        || lock.clone(),
        move |_, lock| {
            let start_time = Instant::now();

            for _ in 0..iters_per_thread {
                lock.lock(|_| {});
            }

            start_time.elapsed()
        },
    )
}

/// A bare minimum test which times how long it takes each threat to perform `iters_per_thread`
/// locks/unlocks. Upon locking it also performs an assignment and wrapping add to discourage
/// optimization of the workload on systems that do not completely support `criterion::black_box`.
/// These actions are included in the benchmark time but are not expected to exceed the time
/// required to lock/unlock the mutex.
#[inline(never)]
fn run_no_load_iters<M: Mutex<usize> + Send + Sync + 'static>(
    num_threads: usize,
    iters_per_thread: usize,
) -> Duration {
    let lock = Arc::new(M::new(1usize));

    let lock_time = bench_threads(
        num_threads,
        || lock.clone(),
        move |i, lock| {
            let start_time = Instant::now();

            for _ in 0..iters_per_thread {
                lock.lock(|shared_value| {
                    // Surround in black_box to prevent optimization of workload
                    // Perform a single add which mutates the state to prevent shared_value from
                    // being optimized away in the event of black_box failing on some targets
                    *shared_value = black_box(shared_value.wrapping_add(i));
                });
            }

            start_time.elapsed()
        },
    );

    // Verify the value in the mutex to further discourage optimizations
    assert_eq!(
        lock.lock(|x| *x),
        1 + iters_per_thread * num_threads * (num_threads + 1) / 2
    );

    lock_time
}

/// A small workload which runs 320 rounds of PRBS31. This workload only requires a couple registers
/// to run and is intended to be hard for the compiler to optimize. Inspecting assembly on x86_64
/// shows the compiler just unrolls the loop if it is small enough and not much more.
#[inline(never)]
fn small_workload(x: u32, rounds: usize) -> u32 {
    let mut value = x;
    for _ in 0..rounds {
        let new_bit = ((value >> 30) ^ (value >> 27)) & 1;
        value = ((value << 1) | new_bit) & ((1u32 << 31) - 1);
    }
    value
}

/// A workload that is balanced so the ratio of work inside the exclusive region is
/// `1 / num_threads`.
#[inline(never)]
fn run_balanced_iters<M: Mutex<u32> + Send + Sync + 'static>(
    num_threads: usize,
    iters_per_thread: usize,
) -> Duration {
    let lock = Arc::new(M::new(1u32));

    bench_threads(
        num_threads,
        || lock.clone(),
        move |_, lock| {
            let mut local_value: u32 = 1;
            let mut total_time = Duration::from_secs(0);

            for _ in 0..iters_per_thread {
                let start_time = Instant::now();
                lock.lock(|shared_value| {
                    // Surround in black_box to prevent optimization of workload
                    // Perform a single add which mutates the state to prevent shared_value from
                    // being optimized away in the event of black_box failing on some targets
                    // *shared_value = black_box(shared_value.wrapping_add(i));
                    *shared_value = black_box(small_workload(black_box(*shared_value), 500));
                    local_value = local_value.wrapping_add(*shared_value);
                });
                total_time += start_time.elapsed();

                // Perform proportional work outside of the critical region
                for _ in 0..num_threads - 1 {
                    local_value = black_box(small_workload(black_box(local_value), 500));
                }
            }

            let _ = black_box(local_value);
            total_time
        },
    )
}

/// We need to run each action for extra iterations to ensure the mutex is actually has the chance
/// to be contested.
const ACTIONS_PER_ITER: usize = 1000;

fn run_no_load<M: Mutex<usize> + Send + Sync + 'static, T: Measurement<Value = Duration>>(
    group: &mut BenchmarkGroup<T>,
    threads: usize,
) {
    let bench_id = BenchmarkId::new(M::name(), threads);

    group.bench_function(bench_id, |b| {
        b.iter_custom(|i| {
            let iters = i as usize * ACTIONS_PER_ITER;
            let lock_time = run_true_no_load_iters::<M>(threads, iters);

            // Remove ACTIONS_PER_ITER as a potential factor in the result. The total time will also
            // scale with the number of threads, but that is recorded using
            // `BenchmarkGroup<M>::throughput`.
            lock_time / ACTIONS_PER_ITER as u32
        })
    });
}

fn run_balanced<M: Mutex<u32> + Send + Sync + 'static, T: Measurement<Value = Duration>>(
    group: &mut BenchmarkGroup<T>,
    threads: usize,
) {
    let bench_id = BenchmarkId::new(M::name(), threads);

    group.bench_function(bench_id, |b| {
        b.iter_custom(|i| {
            let iters = i as usize * ACTIONS_PER_ITER;
            let lock_time = run_balanced_iters::<M>(threads, iters);

            // Remove ACTIONS_PER_ITER as a potential factor in the result. The total time will also
            // scale with the number of threads, but that is recorded using
            // `BenchmarkGroup<M>::throughput`.
            lock_time / ACTIONS_PER_ITER as u32
        })
    });
}

/// This test works by running a multiple of the number of threads available on the system.
fn run_over_provisioned(c: &mut Criterion, overflow: usize) {
    let thread_count = overflow * num_cpus::get();

    let mut competing = c.benchmark_group("competing");
    competing.measurement_time(Duration::from_secs(30));
    competing.sample_size(300);

    run_no_load::<std::sync::Mutex<usize>, _>(&mut competing, thread_count);
    run_no_load::<std::sync::RwLock<usize>, _>(&mut competing, thread_count);
    run_no_load::<parking_lot::Mutex<usize>, _>(&mut competing, thread_count);
    run_no_load::<parking_lot::RwLock<usize>, _>(&mut competing, thread_count);
    // run_no_load::<parking_lot::FairMutex<usize>, _>(&mut competing, 1);
    run_no_load::<parking_lot::ReentrantMutex<RefCell<usize>>, _>(&mut competing, thread_count);

    #[cfg(windows)]
    run_no_load::<SrwLock<usize>, _>(&mut competing, thread_count);

    #[cfg(unix)]
    run_no_load::<PthreadMutex<usize>, _>(&mut competing, thread_count);
    competing.finish();
}
fn criterion_benchmark(c: &mut Criterion) {
    let mut uncontested = c.benchmark_group("uncontested");
    uncontested.measurement_time(Duration::from_secs(30));
    uncontested.sample_size(300);

    run_no_load::<std::sync::Mutex<usize>, _>(&mut uncontested, 1);
    run_no_load::<std::sync::RwLock<usize>, _>(&mut uncontested, 1);
    run_no_load::<parking_lot::Mutex<usize>, _>(&mut uncontested, 1);
    run_no_load::<parking_lot::RwLock<usize>, _>(&mut uncontested, 1);
    // run_no_load::<parking_lot::FairMutex<()>, _>(&mut uncontested, 1);
    run_no_load::<parking_lot::ReentrantMutex<RefCell<usize>>, _>(&mut uncontested, 1);

    #[cfg(windows)]
    run_no_load::<SrwLock<usize>, _>(&mut uncontested, 1);

    #[cfg(unix)]
    run_no_load::<PthreadMutex<usize>, _>(&mut uncontested, 1);

    uncontested.finish();

    let mut competing = c.benchmark_group("competing");
    competing.measurement_time(Duration::from_secs(30));
    competing.sample_size(300);

    run_no_load::<std::sync::Mutex<usize>, _>(&mut competing, 2);
    run_no_load::<std::sync::RwLock<usize>, _>(&mut competing, 2);
    run_no_load::<parking_lot::Mutex<usize>, _>(&mut competing, 2);
    run_no_load::<parking_lot::RwLock<usize>, _>(&mut competing, 2);
    // run_no_load::<parking_lot::FairMutex<usize>, _>(&mut competing, 1);
    run_no_load::<parking_lot::ReentrantMutex<RefCell<usize>>, _>(&mut competing, 2);

    #[cfg(windows)]
    run_no_load::<SrwLock<usize>, _>(&mut competing, 2);

    #[cfg(unix)]
    run_no_load::<PthreadMutex<usize>, _>(&mut competing, 2);
    competing.finish();




    let mut no_load = c.benchmark_group("no-load");

    // We need to use logarithmic scaling otherwise it may be hard to see some cases
    no_load.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for threads in 1..=0 {
        // Tell criterion we are bumping up the difficulty so it can account for it in the report
        no_load.throughput(Throughput::Elements(threads as u64));

        run_no_load::<std::sync::Mutex<usize>, _>(&mut no_load, threads);
        run_no_load::<std::sync::RwLock<usize>, _>(&mut no_load, threads);
        run_no_load::<parking_lot::Mutex<usize>, _>(&mut no_load, threads);
        run_no_load::<parking_lot::RwLock<usize>, _>(&mut no_load, threads);
        // run_no_load::<parking_lot::FairMutex<usize>, _>(&mut no_load, threads);
        run_no_load::<parking_lot::ReentrantMutex<RefCell<usize>>, _>(&mut no_load, threads);

        #[cfg(windows)]
        run_no_load::<SrwLock<usize>, _>(&mut no_load, threads);

        #[cfg(unix)]
        run_no_load::<PthreadMutex<usize>, _>(&mut no_load, threads);
    }

    no_load.finish();


    let mut balanced = c.benchmark_group("balanced");
    balanced.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for threads in 1..=0 {
        // Tell criterion we are bumping up the difficulty so it can account for it in the report
        balanced.throughput(Throughput::Elements(threads as u64));

        run_balanced::<std::sync::Mutex<u32>, _>(&mut balanced, threads);
        run_balanced::<std::sync::RwLock<u32>, _>(&mut balanced, threads);
        run_balanced::<parking_lot::Mutex<u32>, _>(&mut balanced, threads);
        run_balanced::<parking_lot::RwLock<u32>, _>(&mut balanced, threads);
        run_balanced::<parking_lot::FairMutex<u32>, _>(&mut balanced, threads);
        run_balanced::<parking_lot::ReentrantMutex<RefCell<u32>>, _>(&mut balanced, threads);

        #[cfg(windows)]
        run_balanced::<SrwLock<u32>, _>(&mut balanced, threads);

        #[cfg(unix)]
        run_balanced::<PthreadMutex<u32>, _>(&mut balanced, threads);

        balanced.bench_function(BenchmarkId::new("workload", threads), |b| {
            b.iter_custom(|i| {
                // FakeMutex is not actually thread safe so it gets run for longer on a single thread
                let iters = i as usize * ACTIONS_PER_ITER * threads;
                let lock_time = run_balanced_iters::<FakeMutex<u32>>(1, iters);
                lock_time / ACTIONS_PER_ITER as u32
            })
        });
    }

    balanced.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
