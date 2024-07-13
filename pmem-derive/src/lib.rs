use proc_macro::TokenStream;
use proc_macro2::Span;
use proc_macro_crate::{crate_name, FoundCrate};
use quote::quote;
use syn::{parse_macro_input, Data::Struct, DeriveInput, Ident};

#[proc_macro_derive(Record)]
pub fn derive_record(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // Extract fields
    let fields = match input.data {
        Struct(data) => data.fields,
        _ => panic!("Record can only be derived for structs"),
    };

    // Generate the sum of the sizes of all fields
    let size_sum = fields.iter().map(|field| {
        let ty = &field.ty;
        quote! {
            std::mem::size_of::<#ty>()
        }
    });

    // Finding the name of the pmem crate at the callsite
    let pmem_crate = crate_name("pmem").expect("pmem is not present in `Cargo.toml`");
    let pmem = match pmem_crate {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!(#ident)
        }
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Generate the impl block
    let name = input.ident;
    let expanded = quote! {
        impl #impl_generics #pmem::memory::Record for #name #ty_generics #where_clause {
            const SIZE: usize = 0 #( + #size_sum )*;

            fn read(data: &[u8]) -> std::result::Result<Self, #pmem::memory::Error> {
                todo!()
            }
            fn write(&self, data: &mut [u8]) -> std::result::Result<(), #pmem::memory::Error> {
                todo!()
            }
        }
    };

    TokenStream::from(expanded)
}
