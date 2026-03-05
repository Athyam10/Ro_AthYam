const int PIN_RED   = 9;
const int PIN_GREEN = 10;
const int PIN_BLUE  = 11;

// Function to set color for common anode RGB LED
void setColor(int R, int G, int B) {
  analogWrite(PIN_RED,   R);
  analogWrite(PIN_GREEN, G);
  analogWrite(PIN_BLUE,  B);
}

void setup() {
  pinMode(PIN_RED,   OUTPUT);
  pinMode(PIN_GREEN, OUTPUT);
  pinMode(PIN_BLUE,  OUTPUT);

  Serial.begin(9600);
  Serial.println("Type 1 = White, 2 = Blue, 3 = Cyan, 4 = Green, 5 = Yellow, 6 = Magenta, 7 = Red, 0 = OFF");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    switch (command) {
      case '1': // White = R + G + B
        setColor(255, 255, 255);
        Serial.println("White ON");
        break;
      case '2': // Blue
        setColor(0, 0, 255);
        Serial.println("Blue ON");
        break;
      case '3': // Cyan = G + B
        setColor(0, 255, 255);
        Serial.println("Cyan ON");
        break;
      case '4': // Green
        setColor(0, 255, 0);
        Serial.println("Green ON");
        break;
      case '5': // Yellow = R + G
        setColor(255, 255, 0);
        Serial.println("Yellow ON");
        break;
      case '6': // Magenta = R + B
        setColor(255, 0, 255);
        Serial.println("Magenta ON");
        break;
      case '7': // Red
        setColor(255, 0, 0);
        Serial.println("Red ON");
        break;
      case '0': // OFF
        setColor(0, 0, 0);
        Serial.println("LED OFF");
        break;
      default:
        Serial.println("Invalid command! Use 0-7.");
        break;
    }
  }
}