use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use proc_macro_crate::{crate_name, FoundCrate};
use quote::{quote, ToTokens};
use syn::{
    parse_macro_input,
    spanned::Spanned,
    Data::{Enum, Struct},
    DataEnum, DataStruct, DeriveInput, Field, Fields, Ident, Index, Type, Variant,
};

#[proc_macro_derive(Record)]
pub fn derive_record(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // Finding the name of the pmem crate at the callsite
    let krate = match crate_name("pmem").expect("pmem is not present in `Cargo.toml`") {
        FoundCrate::Itself => Ident::new("crate", Span::call_site()),
        FoundCrate::Name(name) => Ident::new(&name, Span::call_site()),
    };

    match &input.data {
        Struct(data) => struct_record::derive(&input, data, &krate),
        Enum(data) => enum_record::derive(&input, data, &krate),
        _ => panic!("Record can only be derived for structs and enums"),
    }
}

mod struct_record {
    use super::*;

    pub(super) fn derive(input: &DeriveInput, data: &DataStruct, krate: &Ident) -> TokenStream {
        // Generate the sum of the sizes of all fields
        let fields_size = data.fields.iter().map(|field| {
            let ty = &field.ty;
            quote! { std::mem::size_of::<#ty>() }
        });

        let write_method = write_method(krate, &data.fields);
        let read_method = read_method(krate, &data.fields);

        let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

        // Generate the impl block
        let name = &input.ident;
        let expanded = quote! {
            impl #impl_generics #krate::memory::Record for #name #ty_generics #where_clause {
                const SIZE: usize = #(#fields_size )+*;

                #read_method

                #write_method
            }
        };

        TokenStream::from(expanded)
    }

    fn write_method(krate: &Ident, fields: &Fields) -> TokenStream2 {
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
            fn write(&self, data: &mut [u8]) -> std::result::Result<(), #krate::memory::Error> {
                use #krate::memory::Record;
                let mut offset = 0;
                #(#write_fields)*
                Ok(())
            }
        }
    }

    fn read_method(krate: &Ident, fields: &Fields) -> TokenStream2 {
        // giving all struct members individual names in a form of v0, v1, etc.
        // so that we can work with tuple and named structs in the same way
        let fields_var_and_ty = generate_local_vars(fields);
        let read_expr = fields_var_and_ty.iter().map(|(var, ty)| read_expr(var, ty));
        let fields_vars = fields_var_and_ty.iter().map(|(var, _)| var);

        let struct_init = match fields {
            Fields::Named(fields) => {
                let field_init = fields_vars
                    .zip(fields.named.iter())
                    .map(|(var, field)| (var, field.ident.as_ref().unwrap()))
                    .map(|(var, field)| quote! { #field: #var });
                quote! { Self { #(#field_init),* } }
            }
            Fields::Unnamed(_) => quote! { Self ( #(#fields_vars),* ) },
            Fields::Unit => unimplemented!("Unit structs are not supported"),
        };

        quote! {
            fn read(input: &[u8]) -> std::result::Result<Self, #krate::memory::Error> {
                use #krate::memory::Record;
                let mut offset = 0;
                #(#read_expr)*;
                Ok(#struct_init)
            }
        }
    }
}

mod enum_record {
    use super::*;

    pub(super) fn derive(input: &DeriveInput, data: &DataEnum, krate: &Ident) -> TokenStream {
        let discriminant_type = read_discriminant_ty(input);
        let variants_sizes = data.variants.iter().map(read_variant_size);
        let enum_name = &input.ident;

        let write_expr = data
            .variants
            .iter()
            .map(|v| variant_write_expr(&discriminant_type, &input.ident, v));

        let discriminant_var = Ident::new("discriminant", Span::call_site());
        let discriminant_read_expr = read_expr(&discriminant_var, &discriminant_type);

        let read_expr = data
            .variants
            .iter()
            .map(|v| variant_read_expr(&input.ident, v));

        let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

        let expanded = quote! {
            impl #impl_generics #krate::memory::Record for #enum_name #ty_generics #where_clause {
                const SIZE: usize = std::mem::size_of::<#discriminant_type>() + #krate::memory::max([#(#variants_sizes),*]);

                fn read(input: &[u8]) -> std::result::Result<Self, #krate::memory::Error> {
                    let mut offset = 0;
                    #discriminant_read_expr
                    match (#discriminant_var) {
                        #(#read_expr)*
                        _ => Err(#krate::memory::Error::UnexpectedVariantCode(u64::from(#discriminant_var)))
                    }
                }

                fn write(&self, data: &mut [u8]) -> std::result::Result<(), #krate::memory::Error> {
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

    fn variant_read_expr(enum_name: &Ident, variant: &Variant) -> TokenStream2 {
        let discriminant = read_discriminant(variant);
        let variant_name = &variant.ident;

        let fields_var_and_ty = generate_local_vars(&variant.fields);
        let read_expr = fields_var_and_ty
            .iter()
            .map(|(var_name, ty)| read_expr(var_name, ty))
            .collect::<Vec<_>>();

        let field_vars = fields_var_and_ty
            .into_iter()
            .map(|(var, _)| var)
            .collect::<Vec<_>>();

        // If the variant has no fields, we just return the variant name with no parentheses
        let variant_init = if let Fields::Unit = variant.fields {
            quote! { #enum_name::#variant_name }
        } else {
            quote! { #enum_name::#variant_name(#(#field_vars),*) }
        };

        quote! {
            #discriminant => {
                #(#read_expr)*
                Ok(#variant_init)
            }
        }
    }

    fn variant_write_expr(repr_type: &Ident, enum_name: &Ident, variant: &Variant) -> TokenStream2 {
        let fields_var_and_ty = generate_local_vars(&variant.fields);
        let variant_name = &variant.ident;
        let write_expr = fields_var_and_ty
            .iter()
            .map(|(var, ty)| write_value_expr(ty, &quote! { #var }));
        let fields_vars = fields_var_and_ty.iter().map(|(var, _)| var);

        let discriminant = read_discriminant(variant);
        let discriminant_write_expr = write_value_expr(repr_type, &quote! { #discriminant });
        let variant_match_expr = if let Fields::Unit = variant.fields {
            quote! { #enum_name::#variant_name }
        } else {
            quote! { #enum_name::#variant_name(#(#fields_vars),*) }
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
        let field_sizes = variant.fields.iter().map(|field| {
            let ty = &field.ty;
            quote! { std::mem::size_of::<#ty>() }
        });
        quote! { #(#field_sizes)+* }
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
