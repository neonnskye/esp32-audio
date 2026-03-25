#include <Arduino.h>
#include <WiFiManager.h>
#include <WiFiUdp.h>

WiFiManager wifiManager;
WiFiUDP wifiUDP;

void setup()
{
  Serial.begin(115200);
  delay(1000);

  // wifiManager.resetSettings(); // for debugging
  wifiManager.autoConnect("Amrith's NodeMCU-32S", "pacman@123");

  const int localPort = 5005;
  wifiUDP.begin(localPort);
  Serial.print("Listening on port: ");
  Serial.println(localPort);
}

void loop()
{
  int packetSize = wifiUDP.parsePacket();
  if (packetSize)
  {
    Serial.print("Received packet of size: ");
    Serial.println(packetSize);

    char incoming[4096];
    int len = wifiUDP.read(incoming, 4096);

    if (len > 0)
    {
      incoming[len] = 0;
    }

    Serial.println("Packet received.");
  }
}