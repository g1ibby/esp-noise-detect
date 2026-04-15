use esp_hal::gpio::{Level, Output, OutputConfig, OutputPin};

/// XIAO ESP32S3 specific pin definitions
pub struct XiaoLed {
    pin: Output<'static>,
}

impl XiaoLed {
    /// Create a new XIAO LED controller
    /// The XIAO ESP32S3 user LED is on GPIO21 and uses inverted logic
    /// (LOW = LED ON, HIGH = LED OFF)
    pub fn new(pin: impl OutputPin + 'static) -> Self {
        let output = Output::new(pin, Level::High, OutputConfig::default());

        Self { pin: output }
    }

    /// Turn the LED on (sets GPIO21 LOW for XIAO ESP32S3)
    pub fn on(&mut self) {
        self.pin.set_low();
    }

    /// Turn the LED off (sets GPIO21 HIGH for XIAO ESP32S3)
    pub fn off(&mut self) {
        self.pin.set_high();
    }

    /// Toggle the LED state
    pub fn toggle(&mut self) {
        self.pin.toggle();
    }

    /// Check if LED is currently on
    /// Returns true if LED is on (pin is LOW)
    pub fn is_on(&self) -> bool {
        self.pin.is_set_low()
    }
}
