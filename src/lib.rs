use std::{
    borrow::BorrowMut,
    cell::{RefCell, RefMut},
    fmt::Debug,
    rc::Rc,
};

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
        foo();
    }
}

fn foo() {
    let mut handle: Handle<Foo> = create_entity(Foo { a: 8 });
    let mut entity: Box<dyn JoinHandle> = handle.join_handle();

    handle.as_mut().a = 42;
    drop(handle);
    let entity: Box<dyn ServiceEntity> = entity.join().unwrap();
    entity.is_changed();
}

#[derive(Debug)]
struct Foo {
    a: u8,
}

impl ServiceEntity for Foo {
    fn is_changed(&self) {
        dbg!(self.a);
    }

    fn write(&self, buffer: &[u8]) {
        todo!()
    }
}

fn create_entity<T>(entity: T) -> Handle<T> {
    Handle {
        value: Rc::new(RefCell::new(entity)),
    }
}

struct Handle<T> {
    value: Rc<RefCell<T>>,
}

impl<T> Handle<T> {
    fn as_mut(&mut self) -> RefMut<T> {
        RefCell::borrow_mut(&self.value)
    }
}

impl<T: Debug + ServiceEntity + 'static> Handle<T> {
    fn join_handle(&self) -> Box<dyn JoinHandle> {
        Box::new(HandleImpl(Some(Rc::clone(&self.value))))
    }
}

struct HandleImpl<T>(Option<Rc<RefCell<T>>>);

impl<T: Debug + ServiceEntity + 'static> JoinHandle for HandleImpl<T> {
    fn join(&mut self) -> Option<Box<dyn ServiceEntity>> {
        let v = self.0.take().unwrap();
        let entity = Rc::try_unwrap(v).unwrap().into_inner();
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
