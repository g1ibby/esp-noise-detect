// (1) No backend selected. Catches the silent-fallback footgun in
//     cubecl/build.rs where zero or multiple backends collapse to
//     test_runtime_default = wgpu — fine on a Mac, broken on a CUDA-only box.
#[cfg(not(any(
    feature = "test-metal",
    feature = "test-vulkan",
    feature = "test-cuda",
    feature = "test-cpu",
)))]
compile_error!(
    "choose a test backend: --features test-metal | test-vulkan | test-cuda | test-cpu"
);

// (2) test-cuda on macOS: would try to build cubecl-cuda and fail deep in
//     a build.rs panic. Catch it up front.
#[cfg(all(feature = "test-cuda", target_os = "macos"))]
compile_error!(
    "test-cuda is not supported on macOS — CUDA toolkit is Linux/Windows only. \
     Use --no-default-features --features \"std,test-metal\" instead."
);

// (3) test-metal on non-macOS: MSL shader compiler is unavailable.
#[cfg(all(feature = "test-metal", not(target_os = "macos")))]
compile_error!(
    "test-metal requires macOS. On Linux/NVIDIA use \
     --no-default-features --features \"std,test-cuda\"; \
     on Linux/AMD or as a fallback use --features \"std,test-vulkan\"."
);

mod suite;
