float A = 150.0;       // 振幅（像素）；
float f = 0.30;        // 頻率 (Hz)；
float omega = TWO_PI * f; // 角頻率；
float t = 0;           // 目前時間；
float dt = 1.0 / 30.0;  // 時間步長（30 FPS）；

int wallX = 150;       // 牆壁 x 位置（像素）；
int eqX = 450;         // 平衡位置（質量塊中心，畫面中間偏左）；
int centerY = 175;     // y 中心線（視窗高度一半）；
int massSize = 60;     // 質量塊大小（正方形）；
int springCoils = 28;  // 彈簧圈數；
float springAmp = 25;  // 彈簧上下振幅（像素）；

void setup() {
  size(900, 350);      // 視窗大小；
  frameRate(30);       // 幀率 30 FPS；
}

void draw() {
  background(#bfbfbf); // 灰色背景；
  
  // 計算目前位移（像素單位）
  float displacement = A * cos(omega * t);
  int massCenterX = eqX + (int)displacement;
  
  // 畫牆壁（白色粗線）
  stroke(255);         // 白色；
  strokeWeight(12);    // 粗線；
  line(wallX, centerY - 120, wallX, centerY + 120);
  
  // 畫彈簧（逐段 line 繪製，交替上下）
  stroke(255);         // 彈簧白色；
  strokeWeight(5);     // 彈簧線寬；
  int leftSpring = wallX + 15;                 // 彈簧起點（離牆一點）
  int rightSpring = massCenterX - massSize/2;  // 彈簧終點（質量塊左側）
  float segment = (rightSpring - leftSpring) / (float)(springCoils + 1);
  
  float prevX = leftSpring;
  float prevY = centerY;
  for (int i = 1; i <= springCoils + 1; i++) {
    float currX = leftSpring + i * segment;
    float currY = centerY;
    if (i <= springCoils) {
      currY += (i % 2 == 1 ? springAmp : -springAmp);  // 奇數上、偶數下；
    }
    line(prevX, prevY, currX, currY);
    prevX = currX;
    prevY = currY;
  }
  
  // 畫質量塊（藍色填滿 + 深藍邊框，一次完成）
  fill(0, 0, 255);           // 藍色填滿；
  stroke(0, 0, 128);         // 深藍邊框；
  strokeWeight(4);           // 邊框粗細；
  rectMode(CENTER);
  rect(massCenterX, centerY, massSize, massSize, 12);  // 圓角矩形；
  
  // 時間前進
  t += dt;
  
  // 10 秒後重置（循環播放）
  if (t > 10.0) {
    t = 0;
  }
}
