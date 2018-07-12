
var input, button, greeting;

function setup() {

  // create canvas
  createCanvas(280, 280);
  button = createButton();
  button.position(0, 0);
  button.mousePressed(greet);
  textAlign(CENTER);
  textSize(50);
  nunulinam()
 }

var duomenysIshEkrano = []

function nunulinam(){
    for(var x = 0;x<28;x++){
        duomenysIshEkrano[x] = []
        for(var y = 0;y<28;y++){
            duomenysIshEkrano[x][y] = 0
        }
    }
}
function draw() {
  stroke(2);
  if (mouseIsPressed === true) {
      mouseX_a = parseInt(mouseX/280 * 28,10)
      mouseY_a = parseInt(mouseY/280 * 28,10)
      duomenysIshEkrano[mouseY_a][mouseX_a] = 1;
    

    //duomenysIshEkrano[duomenysIshEkrano.length] = {mouseX, mouseY, pmouseX, pmouseY}
    line(mouseX, mouseY, pmouseX, pmouseY);
  }


}
function greet() {
  postData(JSON.stringify(duomenysIshEkrano))
  console.log(duomenysIshEkrano)
}
