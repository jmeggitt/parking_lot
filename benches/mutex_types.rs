use std::cell::RefCell;
use std::cell::UnsafeCell;

pub trait Mutex<T: 'static + Send>: 'static + Sync + Send {
    const NAME: &'static str;

    fn new(v: T) -> Self;
    fn lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R;
}

impl<T: 'static + Send> Mutex<T> for std::sync::Mutex<T> {
    const NAME: &'static str = "std::sync::Mutex";

    fn new(v: T) -> Self {
        Self::new(v)
    }
    fn lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
        f(&mut *self.lock().unwrap())
    }
}

impl<T: 'static + Send + Sync> Mutex<T> for std::sync::RwLock<T> {
    const NAME: &'static str = "std::sync::RwLock";

    fn new(v: T) -> Self {
        Self::new(v)
    }
    fn lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
        f(&mut *self.write().unwrap())
    }
}

impl<T: 'static + Send> Mutex<T> for parking_lot::Mutex<T> {
    const NAME: &'static str = "parking_lot::Mutex";

    fn new(v: T) -> Self {
        Self::new(v)
    }
    fn lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
        f(&mut *self.lock())
    }
}

impl<T: 'static + Send> Mutex<T> for parking_lot::FairMutex<T> {
    const NAME: &'static str = "parking_lot::FairMutex";

    fn new(v: T) -> Self {
        Self::new(v)
    }
    fn lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
        f(&mut *self.lock())
    }
}

/// As a comparison, also test a RwLock that gets used even though no reads occur.
impl<T: 'static + Send> Mutex<T> for parking_lot::ReentrantMutex<RefCell<T>> {
    const NAME: &'static str = "parking_lot::ReentrantMutex<RefCell>";

    fn new(v: T) -> Self {
        Self::new(RefCell::new(v))
    }
    fn lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
        let lock = self.lock();
        let mut ref_mut = RefCell::borrow_mut(&*lock);
        f(&mut *ref_mut)
    }
}

/// As a comparison, also test a RwLock that gets used even though no reads occur.
impl<T: 'static + Send + Sync> Mutex<T> for parking_lot::RwLock<T> {
    const NAME: &'static str = "parking_lot::RwLock";

    fn new(v: T) -> Self {
        Self::new(v)
    }
    fn lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
        f(&mut *self.write())
    }
}

/// A regular value which pretends to be a mutex. Can be used with regular tests to measure
/// benchmark overhead. However since it does not employ any locking, it should only be used in a
/// single threaded context.
#[repr(transparent)]
pub struct FakeMutex<T>(UnsafeCell<T>);

unsafe impl<T> Sync for FakeMutex<T> {}
unsafe impl<T> Send for FakeMutex<T> {}

impl<T: 'static + Send> Mutex<T> for FakeMutex<T> {
    const NAME: &'static str = "workload";

    fn new(v: T) -> Self {
        FakeMutex(UnsafeCell::new(v))
    }

    fn lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
        unsafe { f(&mut *self.0.get()) }
    }
}

#[cfg(windows)]
pub use windows::*;

#[cfg(windows)]
mod windows {
    use super::*;

    use winapi::um::synchapi;

    pub struct SrwLock<T>(UnsafeCell<T>, UnsafeCell<synchapi::SRWLOCK>);

    unsafe impl<T> Sync for SrwLock<T> {}
    unsafe impl<T: Send> Send for SrwLock<T> {}

    impl<T: 'static + Send> Mutex<T> for SrwLock<T> {
        const NAME: &'static str = "winapi_srwlock";

        fn new(v: T) -> Self {
            let mut h: synchapi::SRWLOCK = synchapi::SRWLOCK {
                Ptr: std::ptr::null_mut(),
            };

            unsafe {
                synchapi::InitializeSRWLock(&mut h);
            }
            SrwLock(UnsafeCell::new(v), UnsafeCell::new(h))
        }
        fn lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
            unsafe {
                synchapi::AcquireSRWLockExclusive(self.1.get());
                let res = f(&mut *self.0.get());
                synchapi::ReleaseSRWLockExclusive(self.1.get());
                res
            }
        }
    }
}

#[cfg(unix)]
pub use unix::*;

#[cfg(unix)]
mod unix {
    use super::*;

    pub struct PthreadMutex<T>(UnsafeCell<T>, UnsafeCell<libc::pthread_mutex_t>);

    unsafe impl<T> Sync for PthreadMutex<T> {}

    impl<T: 'static + Send> Mutex<T> for PthreadMutex<T> {
        const NAME: &'static str = "pthread_mutex_t";

        fn new(v: T) -> Self {
            PthreadMutex(
                UnsafeCell::new(v),
                UnsafeCell::new(libc::PTHREAD_MUTEX_INITIALIZER),
            )
        }
        fn lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
            unsafe {
                libc::pthread_mutex_lock(self.1.get());
                let res = f(&mut *self.0.get());
                libc::pthread_mutex_unlock(self.1.get());
                res
            }
        }
    }

    impl<T> Drop for PthreadMutex<T> {
        fn drop(&mut self) {
            unsafe {
                libc::pthread_mutex_destroy(self.1.get());
            }
        }
    }
}
