mod memory;
mod page;

pub use memory::{Handle, Memory, Ptr, Storable, Transaction};
pub use page::Addr;

#[macro_export]
macro_rules! ensure {
    ($predicate:expr, $error:expr) => {
        if !($predicate) {
            return Err($error);
        }
    };
}
