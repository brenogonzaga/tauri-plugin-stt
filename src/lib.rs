use tauri::{
    plugin::{Builder, TauriPlugin},
    Manager, Runtime,
};

pub use models::*;

#[cfg(desktop)]
mod desktop;
#[cfg(mobile)]
mod mobile;

mod commands;
mod error;
mod models;
mod paths;

pub use error::{Error, Result};
pub use paths::{
    get_model_path, get_models_dir, list_available_models, model_exists, validate_path,
};

#[cfg(desktop)]
use desktop::Stt;
#[cfg(mobile)]
use mobile::Stt;

/// Extensions to [`tauri::App`], [`tauri::AppHandle`] and [`tauri::Window`] to access the stt APIs.
pub trait SttExt<R: Runtime> {
    fn stt(&self) -> &Stt<R>;
}

impl<R: Runtime, T: Manager<R>> crate::SttExt<R> for T {
    fn stt(&self) -> &Stt<R> {
        self.state::<Stt<R>>().inner()
    }
}

/// Initializes the plugin.
pub fn init<R: Runtime>() -> TauriPlugin<R> {
    let mut builder = Builder::new("stt");

    #[cfg(desktop)]
    {
        builder = builder.invoke_handler(tauri::generate_handler![
            commands::start_listening,
            commands::stop_listening,
            commands::is_available,
            commands::get_supported_languages,
            commands::check_permission,
            commands::request_permission,
            commands::register_listener,
            commands::remove_listener,
        ]);
    }

    #[cfg(mobile)]
    {
        builder = builder.invoke_handler(tauri::generate_handler![
            commands::start_listening,
            commands::stop_listening,
            commands::is_available,
            commands::get_supported_languages,
            commands::check_permission,
            commands::request_permission,
        ]);
    }

    builder
        .setup(|app, api| {
            #[cfg(mobile)]
            let stt = mobile::init(app, api)?;
            #[cfg(desktop)]
            let stt = desktop::init(app, api)?;
            app.manage(stt);
            Ok(())
        })
        .build()
}
