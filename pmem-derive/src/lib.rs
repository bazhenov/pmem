use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use proc_macro_crate::{crate_name, FoundCrate};
use quote::{quote, ToTokens};
use syn::{
    parse_macro_input,
    spanned::Spanned,
    Data::{Enum, Struct},
    DataEnum, DataStruct, DeriveInput, Field, Fields, FieldsNamed, Ident, Index, Type, Variant,
};

/// Implements the [Record] trait for structs and enums
#[proc_macro_derive(Record)]
pub fn derive_record(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // Finding the name of the pmem crate at the callsite
    let pmem = match crate_name("pmem").expect("pmem is not present in `Cargo.toml`") {
        FoundCrate::Itself => Ident::new("crate", Span::call_site()),
        FoundCrate::Name(name) => Ident::new(&name, Span::call_site()),
    };

    match &input.data {
        Struct(data) => struct_record::derive(&input, data, &pmem),
        Enum(data) => enum_record::derive(&input, data, &pmem),
        _ => panic!("Record can only be derived for structs and enums"),
    }
}

mod struct_record {
    use super::*;

    pub(super) fn derive(input: &DeriveInput, data: &DataStruct, pmem: &Ident) -> TokenStream {
        // Generate the sum of the sizes of all fields
        let fields_size = data.fields.iter().map(|field| {
            let ty = &field.ty;
            quote! { std::mem::size_of::<#ty>() }
        });

        let write_method = write_method(pmem, &data.fields);
        let read_method = read_method(pmem, &data.fields);

        let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

        // Generate the impl block
        let name = &input.ident;
        let expanded = quote! {
            impl #impl_generics #pmem::memory::Record for #name #ty_generics #where_clause {
                const SIZE: usize = #(#fields_size )+*;

                #read_method

                #write_method
            }
        };

        TokenStream::from(expanded)
    }

    fn write_method(pmem: &Ident, fields: &Fields) -> TokenStream2 {
        let write_fields = fields.iter().enumerate().map(|(idx, field)| {
            let access = match &field.ident {
                Some(ident) => quote! { self.#ident },
                None => {
                    let idx = Index::from(idx);
                    quote! { self.#idx }
                }
            };
            write_value_expr(&field.ty, &access)
        });

        quote! {
            fn write(&self, data: &mut [u8]) -> std::result::Result<(), #pmem::memory::Error> {
                use #pmem::memory::Record;
                let mut offset = 0;
                #(#write_fields)*
                Ok(())
            }
        }
    }

    fn read_method(pmem: &Ident, fields: &Fields) -> TokenStream2 {
        // giving all struct members individual names in a form of v0, v1, etc.
        // so that we can work with tuple and named structs in the same way
        let fields_var_and_ty = generate_local_vars(fields);
        let read_expr = fields_var_and_ty.iter().map(|(var, ty)| read_expr(var, ty));
        let fields_vars = fields_var_and_ty
            .iter()
            .map(|(var, _)| var)
            .collect::<Vec<_>>();

        let struct_init = match fields {
            Fields::Named(fields) => named_init_expr(&fields_vars, fields, None),
            Fields::Unnamed(_) => unnamed_init_expr(&fields_vars, None),
            Fields::Unit => unimplemented!("Unit structs are not supported"),
        };

        quote! {
            fn read(input: &[u8]) -> std::result::Result<Self, #pmem::memory::Error> {
                use #pmem::memory::Record;
                let mut offset = 0;
                #(#read_expr)*;
                Ok(#struct_init)
            }
        }
    }
}

mod enum_record {
    use super::*;

    pub(super) fn derive(input: &DeriveInput, data: &DataEnum, pmem: &Ident) -> TokenStream {
        let discriminant_ty = read_discriminant_ty(input);
        let variants_sizes = data.variants.iter().map(read_variant_size);
        let enum_name = &input.ident;

        let write_expr = data
            .variants
            .iter()
            .map(|v| variant_write_expr(&discriminant_ty, v));

        let discriminant_var = Ident::new("discriminant", Span::call_site());
        let discriminant_read_expr = read_expr(&discriminant_var, &discriminant_ty);

        let read_expr = data.variants.iter().map(variant_read_expr);

        let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

        let expanded = quote! {
            impl #impl_generics #pmem::memory::Record for #enum_name #ty_generics #where_clause {
                const SIZE: usize = std::mem::size_of::<#discriminant_ty>() + #pmem::memory::max([#(#variants_sizes),*]);

                fn read(input: &[u8]) -> std::result::Result<Self, #pmem::memory::Error> {
                    use #pmem::memory::Record;
                    let mut offset = 0;
                    #discriminant_read_expr
                    match (#discriminant_var) {
                        #(#read_expr)*
                        _ => Err(#pmem::memory::Error::UnexpectedVariantCode(u64::from(#discriminant_var)))
                    }
                }

                fn write(&self, data: &mut [u8]) -> std::result::Result<(), #pmem::memory::Error> {
                    use #pmem::memory::Record;
                    match self {
                        #(#write_expr)*
                    }
                    Ok(())
                }
            }
        };

        TokenStream::from(expanded)
    }

    /// Reads the type given in `#[repr]` attribute
    fn read_discriminant_ty(input: &DeriveInput) -> Ident {
        fn try_read_repr(attr: &syn::Attribute) -> Option<Ident> {
            attr.path()
                .is_ident("repr")
                .then(|| attr.parse_args::<Ident>().ok())
                .flatten()
        }

        input
            .attrs
            .iter()
            .find_map(try_read_repr)
            .expect("#[repr] attribute should be defined on enum")
    }

    fn variant_read_expr(variant: &Variant) -> TokenStream2 {
        let discriminant = read_discriminant(variant);
        let variant_name = &variant.ident;

        let fields_var_and_ty = generate_local_vars(&variant.fields);
        let read_expr = fields_var_and_ty
            .iter()
            .map(|(var_name, ty)| read_expr(var_name, ty))
            .collect::<Vec<_>>();

        let field_vars = fields_var_and_ty
            .iter()
            .map(|(var, _)| var)
            .collect::<Vec<_>>();

        let variant_init = match &variant.fields {
            Fields::Unit => quote! { Self::#variant_name },
            Fields::Unnamed(_) => unnamed_init_expr(&field_vars, Some(variant_name)),
            Fields::Named(fields) => named_init_expr(&field_vars, fields, Some(variant_name)),
        };

        quote! {
            #discriminant => {
                #(#read_expr)*
                Ok(#variant_init)
            }
        }
    }

    fn variant_write_expr(repr_type: &Ident, variant: &Variant) -> TokenStream2 {
        let fields_var_and_ty = generate_local_vars(&variant.fields);
        let variant_name = &variant.ident;
        let write_expr = fields_var_and_ty
            .iter()
            .map(|(var, ty)| write_value_expr(ty, &quote! { #var }));
        let fields_vars = fields_var_and_ty
            .iter()
            .map(|(var, _)| var)
            .collect::<Vec<_>>();

        let discriminant = read_discriminant(variant);
        let discriminant_write_expr = write_value_expr(repr_type, &quote! { #discriminant });

        let variant_match_expr = match &variant.fields {
            Fields::Unit => quote! { Self::#variant_name },
            Fields::Unnamed(_) => unnamed_init_expr(&fields_vars, Some(variant_name)),
            Fields::Named(fields) => named_init_expr(&fields_vars, fields, Some(variant_name)),
        };
        quote! {
            #variant_match_expr => {
                let mut offset = 0;
                #discriminant_write_expr
                #(#write_expr)*
            },
        }
    }

    /// Reads the discriminant value from the variant of the enum
    /// For example, in `A = 1`, `1` is the discriminant
    fn read_discriminant(variant: &Variant) -> &syn::Expr {
        &variant
            .discriminant
            .as_ref()
            .expect("Enum discriminant should be goven")
            .1
    }

    /// Builds the expression of the sum of all variant fields' sizes
    fn read_variant_size(variant: &syn::Variant) -> TokenStream2 {
        let field_sizes = variant
            .fields
            .iter()
            .map(|field| {
                let ty = &field.ty;
                quote! { std::mem::size_of::<#ty>() }
            })
            .collect::<Vec<_>>();
        if field_sizes.is_empty() {
            quote! { 0 }
        } else {
            quote! { #(#field_sizes)+* }
        }
    }
}

/// Create a local variables (v0, v1, ...) if the fields are unnamed
/// or use the field name if it is named with v prefix
fn generate_local_vars(fields: &Fields) -> Vec<(Ident, &Type)> {
    fn create_new_var(idx: usize, field: &Field) -> Ident {
        let var_name = field
            .ident
            .as_ref()
            .map_or_else(|| format!("v{}", idx), |ident| format!("v{}", ident));
        Ident::new(&var_name, field.ident.span())
    }

    fields
        .iter()
        .enumerate()
        .map(|(idx, field)| (create_new_var(idx, field), &field.ty))
        .collect::<Vec<_>>()
}

fn write_value_expr(ty: &impl ToTokens, access: &TokenStream2) -> TokenStream2 {
    quote! {
        let len = <#ty as Record>::SIZE;
        <#ty as Record>::write(&#access, &mut data[offset..offset + len])?;
        offset += len;
    }
}

fn read_expr(var_name: &Ident, ty: &impl ToTokens) -> TokenStream2 {
    quote! {
        let len = <#ty as Record>::SIZE;
        let #var_name = <#ty as Record>::read(&input[offset..offset + len])?;
        offset += len;
    }
}

/// Generate the expression to initialize the named fields expression for a struct or tuple
/// eg. `Self { field0: v0, field1: v1, ... }` or `Self::Variant { field0: v0, field1: v1, ... }`
///
/// `variant_name` is the name of the variant in case of enum and always `None` for struct.
fn named_init_expr(
    fields_vars: &[&Ident],
    fields: &FieldsNamed,
    variant_name: Option<&Ident>,
) -> TokenStream2 {
    let field_init = fields_vars
        .iter()
        .zip(fields.named.iter())
        .map(|(var, field)| (var, field.ident.as_ref().unwrap()))
        .map(|(var, field)| quote! { #field: #var });
    if let Some(variant_name) = variant_name {
        quote! { Self::#variant_name { #(#field_init),* } }
    } else {
        quote! { Self { #(#field_init),* } }
    }
}

/// Generate the expression to initialize unnamed fields expression for a struct or tuple.
/// eg. `Self (v0, v1, v2)`
///
/// `variant_name` is the name of the variant in case of enum and always `None` for struct.
fn unnamed_init_expr(fields_vars: &[&Ident], variant_name: Option<&Ident>) -> TokenStream2 {
    if let Some(variant_name) = variant_name {
        quote! { Self::#variant_name ( #(#fields_vars),* ) }
    } else {
        quote! { Self ( #(#fields_vars),* ) }
    }
}
