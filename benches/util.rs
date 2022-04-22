use std::sync::{Arc, Barrier};
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;

/// A utility function to perform a multi-threaded benchmark. Threads are created with elevated
/// priority levels to reduce noise due to OS scheduling and use `std::sync::Barrier` to
/// synchronize the start time across all threads. The result of the function is the sum of the
/// `Duration`s emitted by all of the threads.
///
/// When the `critical-benchmark` feature is enabled, the priority of the threads may be increased
/// (platform dependent) to levels which can cause the system to freeze or function in a more
/// limited capacity until the test is completed. See [`THREAD_PRIORITY_TIME_CRITICAL`].
///
/// [`THREAD_PRIORITY_TIME_CRITICAL`]: https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadpriority
pub fn bench_threads<P, F, A>(thread_count: usize, mut prepare: P, execute: F) -> Duration
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

/// A small workload which runs a specified number of rounds of PRBS31. This workload only requires
/// a couple registers and does not make use of any memory. It is intended to be hard for the
/// compiler to optimize while providing a minimal workload. When seeded with a value `1u32`, it
/// takes 2,147,483,646 rounds (verified experimentally) before `1u32` is reached and the cycle
/// repeats.
///
/// See https://en.wikipedia.org/wiki/Pseudorandom_binary_sequence
///
/// As expected, inspecting the assembly produced on an x86_64 machine using rustc 1.60 shows that
/// the compiler is unable to apply any meaningful optimizations to the function.
///
/// Given that an uncontested mutex lock can take up to around ~100 CPU cycles and a mutex contested
/// by 2 threads can take up about ~500 CPU cycles, running this function with multiples of ~500
/// rounds should ensure it exceeds the time required to lock a mutex.
pub fn workload(x: u32, rounds: usize) -> u32 {
    let mut value = x;
    for _ in 0..rounds {
        let new_bit = ((value >> 30) ^ (value >> 27)) & 1;
        value = ((value << 1) | new_bit) & ((1u32 << 31) - 1);
    }
    value
}
