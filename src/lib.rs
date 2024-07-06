pub mod fs;
mod memory;
pub mod page;

pub use memory::{
    parse_optional_ptr, write_optional_ptr, Handle, Memory, Ptr, Storable, Transaction,
};
pub use page::Addr;

#[macro_export]
macro_rules! ensure {
    ($predicate:expr, $error:expr) => {
        if !($predicate) {
            return Err($error);
        }
    };
}
