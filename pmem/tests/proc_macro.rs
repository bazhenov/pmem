use pmem::Record;
use pmem_derive::Record;
use std::{fmt::Debug, usize};

#[test]
fn size_of_tuples() {
    #[derive(Record)]
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

    assert_read_write_eq(A { a: 1, b: 2 });
}

#[test]
fn serialization_of_struct_with_array() {
    #[derive(Record, PartialEq, Debug)]
    struct A {
        a: [u8; 4],
    }

    assert_read_write_eq(A { a: [0, 1, 2, 3] });
}

#[test]
fn size_of_enum() {
    #[derive(Record, PartialEq, Debug)]
    #[repr(u8)]
    enum A {
        U16(u16) = 1,
        U32(u32) = 2,
    }

    assert_eq!(A::SIZE, 5);
}

#[test]
fn serialization_of_enum() {
    #[derive(Record, PartialEq, Debug)]
    #[repr(u8)]
    enum A {
        U16(u16) = 1,
        U32(u32) = 2,
        U64(u64) = 3,
    }

    assert_read_write_eq(A::U16(42));
    assert_read_write_eq(A::U32(42));
    assert_read_write_eq(A::U64(42));
}

#[test]
fn serialization_of_enum_with_generics() {
    #[derive(Record, PartialEq, Debug)]
    #[repr(u8)]
    enum Opt<T: Record> {
        Some(T) = 1,
        None = 2,
    }

    assert_eq!(Opt::<u32>::SIZE, 5);
    assert_read_write_eq(Opt::Some(42u32));
}

#[test]
fn serialization_of_enum_with_named_fields() {
    #[derive(Record, PartialEq, Debug)]
    #[repr(u8)]
    enum Path {
        Windows { drive: u8, inode: u64 } = 1,
        Unix(u64) = 2,
    }

    assert_read_write_eq(Path::Windows { drive: 1, inode: 2 });
    assert_read_write_eq(Path::Unix(42));
}

fn assert_read_write_eq<T: Record + Debug + PartialEq>(a: T) {
    let mut buffer = vec![0; T::SIZE];
    a.write(buffer.as_mut()).unwrap();
    let a_copy = T::read(buffer.as_slice()).unwrap();

    assert_eq!(a, a_copy);
}
