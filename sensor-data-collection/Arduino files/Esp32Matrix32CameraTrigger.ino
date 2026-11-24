#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include "driver/gpio.h"

// =====================================================
// ESP32 Feather V1 + 32x32 velostat matrix scanner
// Row select: 2x CD74HC154M96
// Col select: ADG732 (CS tied LOW, always enabled)
// Pot control: MCP4018 (I2C)
// ADC: AD7466 (SPI, MISO only)
// Protocol: host sends 'w' -> MCU returns one full frame
//
// Camera trigger:
//   GPIO13 -> trigger driver/input, wired in parallel to the OPTOIN (Line0)
//   of every camera. One rising-edge pulse is emitted per frame, after the
//   whole matrix is sampled and before the frame is sent, so every 'w' from
//   the host produces exactly one tactile frame and one camera trigger, in
//   that order. The host therefore sets the trigger rate, and frame N of
//   the serial stream belongs to image N of every camera.
//   The serial protocol is unchanged, so every existing host tool still works.
//
//   Do NOT move the trigger to GPIO33: that pin is COL_A1 of the ADG732 and
//   toggles 512 times per frame. GPIO13 also drives the Feather's red LED.
// =====================================================

// ===== Tunables =====
#define BAUDRATE            1000000
#define UART_TX_BUF         4096
#define WAIT_CMD            1
#define ROW_SETTLE_US       120
#define COL_SETTLE_US       25
#define ADC_GUARD_US        2
#define AD7466_SPI_HZ       1000000
#define ADC_AVG2            0

// 1: 1 byte / pixel (1024 bytes/frame). 0: little-endian uint16, 12 bits used (2048 bytes/frame)
#define OUTPUT_U8_FRAME     0

// ===== Camera trigger =====
#define CAM_TRIG            GPIO_NUM_13
#define CAM_TRIG_PULSE_US   100

// ===== MCP4018 =====
static const uint8_t MCP4018_ADDR  = 0x2F;
static const int     MCP4018_STEPS = 128;   // valid step/code: 0..127
static const uint8_t MCP4018_STEP  = 100;   // <<< set desired step here

static inline bool mcp4018_set_step(uint8_t step) {
  if (step >= MCP4018_STEPS) step = MCP4018_STEPS - 1;  // clamp to 127
  Wire.beginTransmission(MCP4018_ADDR);
  Wire.write(step);
  return Wire.endTransmission() == 0;
}

// Optional: keep this helper if you still want percent sometimes
static inline bool mcp4018_set_percent(float pct) {
  if (pct < 0.0f) pct = 0.0f;
  if (pct > 1.0f) pct = 1.0f;
  uint8_t step = (uint8_t)roundf(pct * (MCP4018_STEPS - 1));
  return mcp4018_set_step(step);
}

// ===== Rows: CD74HC154M96 =====
// MCU GPIO26,25,16,17 -> A0..A3
// MCU GPIO21: 0 => chip #1 active, 1 => chip #2 active
#define ROW_A0       GPIO_NUM_26
#define ROW_A1       GPIO_NUM_25
#define ROW_A2       GPIO_NUM_16
#define ROW_A3       GPIO_NUM_17
#define ROW_BANK_SEL GPIO_NUM_21

// ===== Cols: ADG732 =====
// A0..A4 = GPIO27,33,15,32,14
#define COL_A0 GPIO_NUM_27
#define COL_A1 GPIO_NUM_33
#define COL_A2 GPIO_NUM_15
#define COL_A3 GPIO_NUM_32
#define COL_A4 GPIO_NUM_14

// ===== AD7466 SPI pins =====
static const int ADC_MISO_PIN = 19;
static const int ADC_SCK_PIN  = 5;
static const int ADC_CS_PIN   = 4;

SPIClass adcSpi(VSPI);

#if OUTPUT_U8_FRAME
static uint8_t frame[32 * 32];
#else
static uint16_t frame[32 * 32];
#endif

static inline void cfgOut(gpio_num_t pin) {
  gpio_config_t c{};
  c.intr_type = GPIO_INTR_DISABLE;
  c.mode = GPIO_MODE_OUTPUT;
  c.pin_bit_mask = (1ULL << pin);
  c.pull_down_en = GPIO_PULLDOWN_DISABLE;
  c.pull_up_en = GPIO_PULLUP_DISABLE;
  gpio_config(&c);
}

static inline void setRowAddr(uint8_t r0_15) {
  gpio_set_level(ROW_A0, (r0_15 >> 0) & 1);
  gpio_set_level(ROW_A1, (r0_15 >> 1) & 1);
  gpio_set_level(ROW_A2, (r0_15 >> 2) & 1);
  gpio_set_level(ROW_A3, (r0_15 >> 3) & 1);
}

static inline void selectRowBank(uint8_t bank) {
  // bank=0 -> first CD74HC154 active, bank=1 -> second active
  gpio_set_level(ROW_BANK_SEL, bank ? 1 : 0);
}

static inline void setColAddr5(uint8_t v5) {
  gpio_set_level(COL_A0, (v5 >> 0) & 1);
  gpio_set_level(COL_A1, (v5 >> 1) & 1);
  gpio_set_level(COL_A2, (v5 >> 2) & 1);
  gpio_set_level(COL_A3, (v5 >> 3) & 1);
  gpio_set_level(COL_A4, (v5 >> 4) & 1);
}

static inline void pulseCameraTrigger() {
  gpio_set_level(CAM_TRIG, 1);
  delayMicroseconds(CAM_TRIG_PULSE_US);
  gpio_set_level(CAM_TRIG, 0);
}

static inline uint16_t ad7466_read_raw12() {
  uint16_t raw;
  adcSpi.beginTransaction(SPISettings(AD7466_SPI_HZ, MSBFIRST, SPI_MODE0));
  digitalWrite(ADC_CS_PIN, LOW);
  raw = adcSpi.transfer16(0x0000);
  digitalWrite(ADC_CS_PIN, HIGH);
  adcSpi.endTransaction();

  // Use raw16[11:0] as 12-bit ADC value
  return raw & 0x0FFF;
}

static inline uint8_t readPixel8() {
  if (ADC_GUARD_US) delayMicroseconds(ADC_GUARD_US);
  (void)ad7466_read_raw12();  // throw-away sample after mux switch
#if ADC_AVG2
  uint16_t a = ad7466_read_raw12();
  uint16_t b = ad7466_read_raw12();
  uint16_t m = (a + b) >> 1;
  return (uint8_t)(m >> 4);
#else
  uint16_t a = ad7466_read_raw12();
  return (uint8_t)(a >> 4);
#endif
}

static inline void wait_command_if_needed() {
#if WAIT_CMD
  while (true) {
    if (Serial.available()) {
      int ch = Serial.read();
      if (ch == 'w') break;
    }
    delay(1);
  }
#endif
}

void setup() {
  // Hold the trigger LOW before anything else so the cameras see no edge at boot
  cfgOut(CAM_TRIG);
  gpio_set_level(CAM_TRIG, 0);

  Serial.setTxBufferSize(UART_TX_BUF);
  Serial.begin(BAUDRATE);

  Wire.begin(23, 22);
  Wire.setClock(400000);

  // Set MCP4018 directly by step/code (0..127)
  (void)mcp4018_set_step(MCP4018_STEP);

  cfgOut(ROW_A0); cfgOut(ROW_A1); cfgOut(ROW_A2); cfgOut(ROW_A3); cfgOut(ROW_BANK_SEL);
  cfgOut(COL_A0); cfgOut(COL_A1); cfgOut(COL_A2); cfgOut(COL_A3); cfgOut(COL_A4);

  pinMode(ADC_CS_PIN, OUTPUT);
  digitalWrite(ADC_CS_PIN, HIGH);
  adcSpi.begin(ADC_SCK_PIN, ADC_MISO_PIN, -1, ADC_CS_PIN);

  gpio_set_level(ROW_BANK_SEL, 0);
  setRowAddr(0);
  setColAddr5(0);
}

void loop() {
  wait_command_if_needed();

  size_t idx = 0;
  for (uint8_t r = 0; r < 32; ++r) {
    selectRowBank(r >= 16);
    setRowAddr(r & 0x0F);
    delayMicroseconds(ROW_SETTLE_US);

    for (uint8_t c = 0; c < 32; ++c) {
      setColAddr5(c);
      delayMicroseconds(COL_SETTLE_US);

#if OUTPUT_U8_FRAME
      uint8_t pix8 = readPixel8();
      frame[idx++] = pix8;
#else
      if (ADC_GUARD_US) delayMicroseconds(ADC_GUARD_US);
      (void)ad7466_read_raw12();
      uint16_t a12 = ad7466_read_raw12();
      frame[idx++] = a12;
#endif
    }
  }

  // The 4x4 sensor sits on the last rows scanned (28-31), so a trigger here is
  // only a few milliseconds after its pads were sampled.
  pulseCameraTrigger();

#if OUTPUT_U8_FRAME
  Serial.write(frame, sizeof(frame)); // 1024 bytes
#else
  Serial.write((uint8_t*)frame, sizeof(frame)); // 2048 bytes (little-endian uint16)
#endif
  Serial.write('\n');
}
