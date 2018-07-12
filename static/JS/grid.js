function setup() {
  createCanvas(280, 280);
  background(255);
}

function draw() {
  stroke(1);
  if (mouseIsPressed === true) {
    line(mouseX, mouseY, pmouseX, pmouseY);
  }
}
