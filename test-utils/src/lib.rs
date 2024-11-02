/// Used for occasional manual debugging of tests. Just throw it to the start of the test
/// to initialize tracing subscriber and use `RUST_LOG` env var to control verbosity.
pub fn init_tracing() {
    use std::io;
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

    let fmt_layer = tracing_subscriber::fmt::layer().with_writer(io::stderr);
    let filter_layer = EnvFilter::from_default_env();

    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(filter_layer)
        .init();
}
