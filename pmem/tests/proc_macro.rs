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

#[test]
fn serialization_of_struct_with_array() {
    #[derive(Record, PartialEq, Debug)]
    struct A {
        a: [u8; 4],
    }

    let a = A { a: [0, 1, 2, 3] };
    let mut buffer = [0; 4];
    a.write(buffer.as_mut()).unwrap();

    let a_copy = A::read(buffer.as_slice()).unwrap();

    assert_eq!(a, a_copy);
}

// #[test]
// fn serialization_of_enum() {
//     #[derive(Record, PartialEq, Debug)]
//     #[repr(u8)]
//     enum A {
//         U16(u16) = 1,
//         U32(u32) = 2,
//     }
//     assert_eq!(A::SIZE, 5);
// }
