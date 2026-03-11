/*
   RGB Rover Status Lighting
   Controls 20 RGB LEDs using MOSFETs

   Pin 9  -> RED MOSFET Gate
   Pin 10 -> GREEN MOSFET Gate
   Pin 11 -> BLUE MOSFET Gate

   Color Logic
   WHITE = Rover stopped
   BLUE  = Rover moving
*/

const int PIN_RED   = 9;
const int PIN_GREEN = 10;
const int PIN_BLUE  = 11;

int linearRPM = 0;

// Function to set RGB color
void setColor(int r, int g, int b)
{
  analogWrite(PIN_RED, r);
  analogWrite(PIN_GREEN, g);
  analogWrite(PIN_BLUE, b);
}

void setup()
{
  pinMode(PIN_RED, OUTPUT);
  pinMode(PIN_GREEN, OUTPUT);
  pinMode(PIN_BLUE, OUTPUT);

  Serial.begin(115200);

  // Start with WHITE (safe mode)
  setColor(255,255,255);
}

void loop()
{
  // Check if command arrives
  if (Serial.available())
  {
    String command = Serial.readStringUntil('\n');

    int firstComma = command.indexOf(',');

    if(firstComma > 0)
    {
      linearRPM = command.substring(0, firstComma).toInt();
    }
  }

  // LED Logic
  if (linearRPM == 0)
  {
    // WHITE = stopped
    setColor(255,255,255);
  }
  else
  {
    // BLUE = moving
    setColor(0,0,255);
  }
}