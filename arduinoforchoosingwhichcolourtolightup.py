const int PIN_RED   = 9;
const int PIN_GREEN = 10;
const int PIN_BLUE  = 11;

void setColor(int R, int G, int B) {
  analogWrite(PIN_RED,   R);
  analogWrite(PIN_GREEN, G);
  analogWrite(PIN_BLUE,  B);
}

void setup() {
  pinMode(PIN_RED,   OUTPUT);
  pinMode(PIN_GREEN, OUTPUT);
  pinMode(PIN_BLUE,  OUTPUT);

  Serial.begin(9600);  // Start serial communication
  Serial.println("Type 1 = Blue, 2 = Green, 3 = Red");
}

void loop() {

  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == '1') {
      setColor(0, 0, 255);   // Blue
      Serial.println("Blue ON");
    }
    else if (command == '2') {
      setColor(0, 255, 0);   // Green
      Serial.println("Green ON");
    }
    else if (command == '3') {
      setColor(255, 0, 0);   // Red
      Serial.println("Red ON");
    }
  }
}