#include <FastLED.h>

#define LED_PIN     11
#define NUM_LEDS    48

CRGB leds[NUM_LEDS];

const int stepsPerRevolution = 200;  // change this to fit the number of steps per revolution
// for your motor

int incomingByte = 0;

const int dirPin = 3;
const int stepPin = 4;
const int enPin = 5;


int reading;




void setup() {
  Serial.begin(9600);
  
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(enPin, OUTPUT);
  digitalWrite(enPin, LOW);

  FastLED.addLeds<WS2812, LED_PIN, GRB>(leds, NUM_LEDS);

}

void loop() {
  ledWhite();
  if (Serial.available()) {
    incomingByte = Serial.parseInt(); // read the incoming byte:
    Serial.print(" I received:");
    //if hell receive 1
    //if banana receive 2
    Serial.println(incomingByte);
    if (incomingByte == 2) {
      ledRed();
      Rotate(true, 400);
      delay(1500);
      Rotate(false, 400);
      ledWhite();
    } else if (incomingByte == 1) {
      ledGreen();
      Rotate(false, 400);
      delay(1500);
      Rotate(true, 400);
      ledWhite();
    }else{
      Rotate(true,2000);
    }
  }
}

//true clockwise false counterclockwise
void Rotate(bool direction, int steps) {
  if (direction) {
    digitalWrite(dirPin, HIGH);
  } else {
    digitalWrite(dirPin, LOW);
  }

  for (int j = 0; j < (steps / 1600); j++) {
    for (int x = 0; x < 1600; x++) {
      digitalWrite(stepPin, HIGH);
      delayMicroseconds(500);
      digitalWrite(stepPin, LOW);
      delayMicroseconds(500);
    }
  }
  //mod
  for (int j = 0; j <= (steps % 1600); j++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(500);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(500);
  }
}

void Stop() {
  digitalWrite(enPin, LOW);
}

void ledWhite() {
  for (int i = 0; i <= NUM_LEDS; i++) {
    leds[i] = CRGB ( 255, 255, 255);
    FastLED.show();
  }
}

void ledRed() {
  for (int i = 0; i <= NUM_LEDS; i++) {
    leds[i] = CRGB ( 255, 0, 0);
    FastLED.show();
  }
}


void ledGreen() {
  for (int i = 0; i <= NUM_LEDS; i++) {
    leds[i] = CRGB ( 0, 255, 0);
    FastLED.show();
  }
}
