const COMMANDS: &[&str] = &[
    "start_listening",
    "stop_listening",
    "is_available",
    "get_supported_languages",
    "check_permission",
    "request_permission",
];

fn main() {
    tauri_plugin::Builder::new(COMMANDS)
        .android_path("android")
        .ios_path("ios")
        .build();
}
