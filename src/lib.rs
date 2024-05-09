use std::{fmt::Debug, rc::Rc};

// mod allocator;
mod page;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

fn foo() {
    let handle: Handle<Foo> = create_entity(Foo { a: 8 });
    let mut entity: Box<dyn JoinHandle> = handle.join_handle();
    drop(handle);
    let entity: Box<dyn ServiceEntity> = entity.join().unwrap();
}

#[derive(Debug)]
struct Foo {
    a: u8,
}

impl ServiceEntity for Foo {
    fn is_changed(&self) {
        todo!()
    }

    fn write(&self, buffer: &[u8]) {
        todo!()
    }
}

fn create_entity<T>(entity: T) -> Handle<T> {
    Handle {
        value: Rc::new(entity),
    }
}

struct Handle<T> {
    value: Rc<T>,
}

impl<T: Debug + ServiceEntity + 'static> Handle<T> {
    fn join_handle(&self) -> Box<dyn JoinHandle> {
        Box::new(HandleImpl(Some(Rc::clone(&self.value))))
    }
}

struct HandleImpl<T>(Option<Rc<T>>);

impl<T: Debug + ServiceEntity + 'static> JoinHandle for HandleImpl<T> {
    fn join(&mut self) -> Option<Box<dyn ServiceEntity>> {
        let v = self.0.take().unwrap();
        let entity = Rc::try_unwrap(v).unwrap();
        Some(Box::new(entity))
    }
}

trait ServiceEntity {
    fn is_changed(&self);
    fn write(&self, buffer: &[u8]);
}

trait JoinHandle {
    fn join(&mut self) -> Option<Box<dyn ServiceEntity>>;
}
