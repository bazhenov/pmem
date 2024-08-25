pub mod driver;
pub mod memory;
pub mod page;

pub use memory::{Handle, Memory, Ptr, Record};

#[macro_export]
macro_rules! ensure {
    ($predicate:expr, $error:expr) => {
        if !($predicate) {
            return Err($error);
        }
    };
}
