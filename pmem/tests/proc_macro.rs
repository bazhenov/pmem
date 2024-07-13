use pmem::Record;
use pmem_derive::Record;

#[test]
fn tuples() {
    #[derive(Record)]
    #[allow(dead_code)]
    struct A(u8, u64);
    assert_eq!(A::SIZE, 9)
}

#[test]
fn structs() {
    #[derive(Record)]
    struct A {
        _a: u8,
        _b: u64,
    }
    assert_eq!(A::SIZE, 9)
}

#[test]
fn generic_structs() {
    #[derive(Record)]
    struct A<T> {
        _a: T,
    }
    assert_eq!(A::<u32>::SIZE, 4)
}
