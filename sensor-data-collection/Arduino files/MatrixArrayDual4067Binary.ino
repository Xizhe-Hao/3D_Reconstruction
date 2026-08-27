#include <Arduino.h>
#include <string.h>

// 16x16 binary scanner using two CD74HC4067 multiplexers.
//
// Row CD74HC4067:
//   COM -> A0 and pulldown resistor
//   S0-S3 -> D4-D7
//   E/#INH -> D8 (LOW = enabled)
//
// Column CD74HC4067:
//   COM -> +5V
//   S0-S3 -> A1-A4
//   E/#INH -> A5 (LOW = enabled)
//
// Camera trigger:
//   D9 -> external trigger driver/input, wired in parallel to the OPTOIN of
//   every camera (the rig currently uses four). One pulse triggers all of
//   them, so adding or removing a camera needs no firmware change.
//   A rising-edge pulse is emitted after all 256 cells are sampled.
//   Each sensor frame is transmitted once before and once after the trigger.
//   Both copies have the same frame index and checksum; the host keeps the
//   first valid copy and discards the duplicate.
//
// Host handshake (newline-terminated ASCII):
//   Arduino repeats M16_READY while idle.
//   Host sends M16_START; Arduino replies M16_ACK and starts acquisition.
//   Host sends M16_PING while running and M16_STOP to return to idle.
//
// Binary frame, little-endian:
//   magic[4]       = "M16B"
//   version        = 1
//   rows           = 16
//   columns        = 16
//   payload type   = 1 (unsigned 8-bit ADC)
//   frame index    = uint32
//   device millis  = uint32
//   payload        = 256 bytes, row-major
//   checksum       = uint16 sum of metadata and payload bytes

#define BAUD_RATE                 500000
#define ROW_COUNT                 16
#define COLUMN_COUNT              16

#define PIN_ADC_INPUT             A0

#define PIN_ROW_MUX_S0            4
#define PIN_ROW_MUX_S1            5
#define PIN_ROW_MUX_S2            6
#define PIN_ROW_MUX_S3            7
#define PIN_ROW_MUX_INHIBIT       8

#define PIN_COLUMN_MUX_S0         A1
#define PIN_COLUMN_MUX_S1         A2
#define PIN_COLUMN_MUX_S2         A3
#define PIN_COLUMN_MUX_S3         A4
#define PIN_COLUMN_MUX_INHIBIT    A5

#define PIN_CAMERA_TRIGGER        9
#define CAMERA_TRIGGER_PULSE_US   100
#define FRAME_PERIOD_US           41667UL  // 24 fps

#define MUX_SETTLE_TIME_US        20
#define READY_INTERVAL_MS         250UL
#define HEARTBEAT_TIMEOUT_MS      2000UL

const byte MAGIC[] = {'M', '1', '6', 'B'};
const char COMMAND_START[] = "M16_START";
const char COMMAND_PING[] = "M16_PING";
const char COMMAND_STOP[] = "M16_STOP";
const char MESSAGE_READY[] = "M16_READY";
const char MESSAGE_ACK[] = "M16_ACK";
const byte PROTOCOL_VERSION = 1;
const byte PAYLOAD_TYPE_ADC_U8 = 1;
const size_t VALUE_COUNT = ROW_COUNT * COLUMN_COUNT;
const size_t FRAME_BYTES = 4 + 1 + 1 + 1 + 1 + 4 + 4 + VALUE_COUNT + 2;

byte frame[FRAME_BYTES];
byte matrixAdc[VALUE_COUNT];
uint32_t frameIndex = 0;
bool acquisitionRunning = false;
unsigned long lastReadySentMillis = 0;
unsigned long lastHostContactMillis = 0;
unsigned long nextFrameStartMicros = 0;
char commandBuffer[24];
uint8_t commandLength = 0;

void setMuxAddress(uint8_t s0, uint8_t s1, uint8_t s2, uint8_t s3,
                   uint8_t channel);
void selectRow(uint8_t row);
void selectColumn(uint8_t column);
void disableMuxes();
void scanMatrix();
void pulseCameraTrigger();
size_t buildBinaryFrame(uint32_t frameMillis);
void processSerialCommands();
void handleCommand(const char *command);
void announceReady();
void enterWaitingState();
void appendByte(byte *buffer, size_t &index, byte value, uint16_t &checksum);
void appendUint16LE(byte *buffer, size_t &index, uint16_t value);
void appendUint32LE(byte *buffer, size_t &index, uint32_t value,
                    uint16_t &checksum);

void setup()
{
  Serial.begin(BAUD_RATE);

  pinMode(PIN_ADC_INPUT, INPUT);

  // Preload HIGH before enabling the output drivers
  digitalWrite(PIN_ROW_MUX_INHIBIT, HIGH);
  digitalWrite(PIN_COLUMN_MUX_INHIBIT, HIGH);
  digitalWrite(PIN_CAMERA_TRIGGER, LOW);
  pinMode(PIN_ROW_MUX_INHIBIT, OUTPUT);
  pinMode(PIN_COLUMN_MUX_INHIBIT, OUTPUT);
  pinMode(PIN_CAMERA_TRIGGER, OUTPUT);

  pinMode(PIN_ROW_MUX_S0, OUTPUT);
  pinMode(PIN_ROW_MUX_S1, OUTPUT);
  pinMode(PIN_ROW_MUX_S2, OUTPUT);
  pinMode(PIN_ROW_MUX_S3, OUTPUT);
  pinMode(PIN_COLUMN_MUX_S0, OUTPUT);
  pinMode(PIN_COLUMN_MUX_S1, OUTPUT);
  pinMode(PIN_COLUMN_MUX_S2, OUTPUT);
  pinMode(PIN_COLUMN_MUX_S3, OUTPUT);

  setMuxAddress(PIN_ROW_MUX_S0, PIN_ROW_MUX_S1,
                PIN_ROW_MUX_S2, PIN_ROW_MUX_S3, 0);
  setMuxAddress(PIN_COLUMN_MUX_S0, PIN_COLUMN_MUX_S1,
                PIN_COLUMN_MUX_S2, PIN_COLUMN_MUX_S3, 0);

  ADCSRA = (ADCSRA & ~(_BV(ADPS2) | _BV(ADPS1) | _BV(ADPS0)))
           | _BV(ADPS2) | _BV(ADPS0);

  enterWaitingState();
}

void loop()
{
  processSerialCommands();

  if (!acquisitionRunning)
  {
    announceReady();
    return;
  }

  if ((unsigned long)(millis() - lastHostContactMillis)
      >= HEARTBEAT_TIMEOUT_MS)
  {
    enterWaitingState();
    return;
  }

  // Keep scan starts on a 24 Hz schedule
  unsigned long nowMicros = micros();
  if ((long)(nowMicros - nextFrameStartMicros) < 0)
  {
    return;
  }
  nextFrameStartMicros += FRAME_PERIOD_US;

  scanMatrix();
  disableMuxes();
  uint32_t frameMillis = millis();

  size_t frameBytes = buildBinaryFrame(frameMillis);
  Serial.write(frame, frameBytes);
  Serial.flush();

  pulseCameraTrigger();

  // Separate the redundant copy from the first USB-serial transaction and
  // from the rising edge that starts the camera transfers
  delayMicroseconds(1000);
  Serial.write(frame, frameBytes);
  frameIndex++;

  if ((long)(micros() - nextFrameStartMicros) >= 0)
  {
    nextFrameStartMicros = micros() + FRAME_PERIOD_US;
  }
}

void scanMatrix()
{
  size_t index = 0;

  for (uint8_t row = 0; row < ROW_COUNT; row++)
  {
    selectRow(row);

    for (uint8_t column = 0; column < COLUMN_COUNT; column++)
    {
      selectColumn(column);
      delayMicroseconds(MUX_SETTLE_TIME_US);

      int rawReading = analogRead(PIN_ADC_INPUT);
      matrixAdc[index++] = (byte)(rawReading >> 2);
    }
  }
}

void selectRow(uint8_t row)
{
  digitalWrite(PIN_ROW_MUX_INHIBIT, HIGH);
  setMuxAddress(PIN_ROW_MUX_S0, PIN_ROW_MUX_S1,
                PIN_ROW_MUX_S2, PIN_ROW_MUX_S3, row);
  digitalWrite(PIN_ROW_MUX_INHIBIT, LOW);
}

void selectColumn(uint8_t column)
{
  digitalWrite(PIN_COLUMN_MUX_INHIBIT, HIGH);
  setMuxAddress(PIN_COLUMN_MUX_S0, PIN_COLUMN_MUX_S1,
                PIN_COLUMN_MUX_S2, PIN_COLUMN_MUX_S3, column);
  digitalWrite(PIN_COLUMN_MUX_INHIBIT, LOW);
}

void setMuxAddress(uint8_t s0, uint8_t s1, uint8_t s2, uint8_t s3,
                   uint8_t channel)
{
  digitalWrite(s0, channel & 0x01);
  digitalWrite(s1, (channel >> 1) & 0x01);
  digitalWrite(s2, (channel >> 2) & 0x01);
  digitalWrite(s3, (channel >> 3) & 0x01);
}

void disableMuxes()
{
  digitalWrite(PIN_ROW_MUX_INHIBIT, HIGH);
  digitalWrite(PIN_COLUMN_MUX_INHIBIT, HIGH);
}

void pulseCameraTrigger()
{
  digitalWrite(PIN_CAMERA_TRIGGER, HIGH);
  delayMicroseconds(CAMERA_TRIGGER_PULSE_US);
  digitalWrite(PIN_CAMERA_TRIGGER, LOW);
}

void processSerialCommands()
{
  while (Serial.available() > 0)
  {
    char incoming = (char)Serial.read();

    if (incoming == '\r')
    {
      continue;
    }

    if (incoming == '\n')
    {
      commandBuffer[commandLength] = '\0';
      if (commandLength > 0)
      {
        handleCommand(commandBuffer);
      }
      commandLength = 0;
      continue;
    }

    if (commandLength < sizeof(commandBuffer) - 1)
    {
      commandBuffer[commandLength++] = incoming;
    }
    else
    {
      commandLength = 0;
    }
  }
}

void handleCommand(const char *command)
{
  if (strcmp(command, COMMAND_START) == 0)
  {
    if (!acquisitionRunning)
    {
      frameIndex = 0;
      lastHostContactMillis = millis();
      Serial.println(MESSAGE_ACK);
      Serial.flush();
      nextFrameStartMicros = micros();
      acquisitionRunning = true;
    }
    return;
  }

  if (strcmp(command, COMMAND_PING) == 0)
  {
    if (acquisitionRunning)
    {
      lastHostContactMillis = millis();
    }
    return;
  }

  if (strcmp(command, COMMAND_STOP) == 0)
  {
    enterWaitingState();
  }
}

void announceReady()
{
  unsigned long now = millis();
  if ((unsigned long)(now - lastReadySentMillis) >= READY_INTERVAL_MS)
  {
    Serial.println(MESSAGE_READY);
    lastReadySentMillis = now;
  }
}

void enterWaitingState()
{
  acquisitionRunning = false;
  disableMuxes();
  digitalWrite(PIN_CAMERA_TRIGGER, LOW);
  commandLength = 0;
  lastReadySentMillis = millis() - READY_INTERVAL_MS;
}

size_t buildBinaryFrame(uint32_t frameMillis)
{
  size_t index = 0;
  uint16_t checksum = 0;

  for (size_t i = 0; i < sizeof(MAGIC); i++)
  {
    frame[index++] = MAGIC[i];
  }

  appendByte(frame, index, PROTOCOL_VERSION, checksum);
  appendByte(frame, index, ROW_COUNT, checksum);
  appendByte(frame, index, COLUMN_COUNT, checksum);
  appendByte(frame, index, PAYLOAD_TYPE_ADC_U8, checksum);
  appendUint32LE(frame, index, frameIndex, checksum);
  appendUint32LE(frame, index, frameMillis, checksum);

  for (size_t i = 0; i < VALUE_COUNT; i++)
  {
    appendByte(frame, index, matrixAdc[i], checksum);
  }

  appendUint16LE(frame, index, checksum);
  return index;
}

void appendByte(byte *buffer, size_t &index, byte value, uint16_t &checksum)
{
  buffer[index++] = value;
  checksum += value;
}

void appendUint16LE(byte *buffer, size_t &index, uint16_t value)
{
  buffer[index++] = value & 0xFF;
  buffer[index++] = (value >> 8) & 0xFF;
}

void appendUint32LE(byte *buffer, size_t &index, uint32_t value,
                    uint16_t &checksum)
{
  appendByte(buffer, index, value & 0xFF, checksum);
  appendByte(buffer, index, (value >> 8) & 0xFF, checksum);
  appendByte(buffer, index, (value >> 16) & 0xFF, checksum);
  appendByte(buffer, index, (value >> 24) & 0xFF, checksum);
}
