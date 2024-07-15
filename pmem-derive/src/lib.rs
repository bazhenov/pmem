use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use proc_macro_crate::{crate_name, FoundCrate};
use quote::quote;
use syn::{
    parse_macro_input, spanned::Spanned, Data::Struct, DeriveInput, Field, Fields, Ident, Index,
};

#[proc_macro_derive(Record)]
pub fn derive_record(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // Extract fields
    let fields = match input.data {
        Struct(data) => data.fields,
        _ => panic!("Record can only be derived for structs"),
    };

    // Finding the name of the pmem crate at the callsite
    let krate = match crate_name("pmem").expect("pmem is not present in `Cargo.toml`") {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!(#ident)
        }
    };

    // Generate the sum of the sizes of all fields
    let size_sum = fields.iter().map(|field| {
        let ty = &field.ty;
        quote! { std::mem::size_of::<#ty>() }
    });

    let write_method = write_method(&krate, &fields);
    let read_method = read_method(&krate, &fields);

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Generate the impl block
    let name = input.ident;
    let expanded = quote! {
        impl #impl_generics #krate::memory::Record for #name #ty_generics #where_clause {
            const SIZE: usize = #(#size_sum )+*;

            #read_method

            #write_method
        }
    };

    TokenStream::from(expanded)
}

fn read_method(krate: &TokenStream2, fields: &Fields) -> TokenStream2 {
    let var_names = fields
        .iter()
        .enumerate()
        .map(|(idx, field)| (field, var_name_for_field(idx, field)))
        .map(|(field, var_name)| Ident::new(&var_name, field.ident.span()))
        .collect::<Vec<_>>();

    let read_instructions = var_names
        .iter()
        .zip(fields.iter())
        .map(|(var_name, field)| {
            let ty = &field.ty;
            quote! {
                let len = <#ty as Record>::SIZE;
                let #var_name = <#ty as Record>::read(&input[offset..offset + len])?;
                offset += len;
            }
        });

    let struct_constructor = match fields {
        Fields::Named(fields) => {
            let field_init = var_names
                .iter()
                .zip(fields.named.iter())
                .map(|(var_name, field)| (var_name, field.ident.as_ref().unwrap()))
                .map(|(var_name, field)| quote! { #field: #var_name });
            quote! { Self { #(#field_init),* } }
        }
        Fields::Unnamed(_) => quote! { Self ( #(#var_names),* ) },
        Fields::Unit => unimplemented!("Unit structs are not supported"),
    };

    quote! {
        fn read(input: &[u8]) -> std::result::Result<Self, #krate::memory::Error> {
            use #krate::memory::Record;
            let mut offset = 0;
            #(#read_instructions)*;
            Ok(#struct_constructor)
        }
    }
}

/// Generate new local variable name for a struct/tuple field
fn var_name_for_field(idx: usize, field: &Field) -> String {
    field
        .ident
        .as_ref()
        .map_or_else(|| format!("v{}", idx), |ident| format!("v{}", ident))
}

fn write_method(krate: &TokenStream2, fields: &Fields) -> TokenStream2 {
    let write_fields = fields.iter().enumerate().map(|(idx, field)| {
        let ty = &field.ty;
        let access = match &field.ident {
            Some(ident) => quote! { self.#ident },
            None => {
                let idx = Index::from(idx);
                quote! { self.#idx }
            }
        };
        quote! {
            let len = <#ty as Record>::SIZE;
            <#ty as Record>::write(&#access, &mut data[offset..offset + len])?;
            offset += len;
        }
    });

    quote! {
        fn write(&self, data: &mut [u8]) -> std::result::Result<(), #krate::memory::Error> {
            use #krate::memory::Record;
            let mut offset = 0;
            #(#write_fields)*
            Ok(())
        }
    }
}
