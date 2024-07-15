use pmem::Record;
use pmem_derive::Record;

#[test]
fn size_of_tuples() {
    #[derive(Record)]
    #[allow(dead_code)]
    struct A(u8, u64);
    assert_eq!(A::SIZE, 9)
}

#[test]
fn size_of_structs() {
    #[derive(Record)]
    struct A {
        _a: u8,
        _b: u64,
    }
    assert_eq!(A::SIZE, 9)
}

#[test]
fn size_of_generic_structs() {
    #[derive(Record)]
    struct A<T: Record> {
        _a: T,
    }
    assert_eq!(A::<u32>::SIZE, 4)
}

#[test]
fn serialization_of_structs() {
    #[derive(Record, PartialEq, Debug)]
    struct A {
        a: u32,
        b: u64,
    }

    let a = A { a: 1, b: 2 };
    let mut buffer = [0; 12];
    a.write(buffer.as_mut()).unwrap();

    let a_copy = A::read(buffer.as_slice()).unwrap();

    assert_eq!(a, a_copy);
}
