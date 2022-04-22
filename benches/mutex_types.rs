use std::cell::RefCell;
use std::cell::UnsafeCell;

pub trait Mutex<T> {
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
impl<T> Mutex<T> for std::sync::RwLock<T> {
    fn new(v: T) -> Self {
        Self::new(v)
    }
    fn lock<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&mut T) -> R,
    {
        f(&mut *self.write().unwrap())
    }
    fn name() -> &'static str {
        "std::sync::RwLock"
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
impl<T> Mutex<T> for parking_lot::ReentrantMutex<RefCell<T>> {
    fn new(v: T) -> Self {
        Self::new(RefCell::new(v))
    }
    fn lock<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&mut T) -> R,
    {
        let lock = self.lock();
        let mut ref_mut = RefCell::borrow_mut(&*lock);
        f(&mut *ref_mut)
    }
    fn name() -> &'static str {
        "parking_lot::ReentrantMutex<RefCell>"
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

/// A regular value which pretends to be a mutex. Can be used with regular tests to measure
/// benchmark overhead. However since it does not employ any locking, it should only be used in a
/// single threaded context.
#[repr(transparent)]
pub struct FakeMutex<T>(UnsafeCell<T>);

unsafe impl<T> Sync for FakeMutex<T> {}
unsafe impl<T> Send for FakeMutex<T> {}

impl<T> Mutex<T> for FakeMutex<T> {
    fn new(v: T) -> Self {
        FakeMutex(UnsafeCell::new(v))
    }

    fn lock<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&mut T) -> R,
    {
        unsafe { f(&mut *self.0.get()) }
    }

    fn name() -> &'static str {
        "workload"
    }
}

#[cfg(windows)]
use winapi::um::synchapi;

#[cfg(windows)]
pub struct SrwLock<T>(UnsafeCell<T>, UnsafeCell<synchapi::SRWLOCK>);
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

#[cfg(unix)]
pub struct PthreadMutex<T>(UnsafeCell<T>, UnsafeCell<libc::pthread_mutex_t>);
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