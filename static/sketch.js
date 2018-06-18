var col; // Current box color
var black, white; // Colors representing black and white
var x, y, w, h; // Box size parameters

function setup() {
	createCanvas(windowWidth, windowHeight);

	white = color(255, 255, 255);
	black = color(0, 0, 0);

	col = genRandColor();
}

function draw() {
	background(127);

	drawRects(col);
}

function genRandColor(){
	return color(random(256), random(256), random(256));
}

// Draw 2 rectangles with color col
function drawRects(col) {
	var bor = 0.05; // Border percentage

	x = width * bor;
	y = height * 2*bor;
	w = width * (1 - 2*bor)/2;
	h = height * (1 - 4*bor);

	strokeWeight(6);
	stroke(20);

	fill(col);
	rect(x, y, w, h);
	rect(x+w, y, w, h);

	// Draw black and white text
	var str = 'The quick brown fox jumped over the lazy dog.';
	textSize(width*bor*1.3);
	noStroke();

	fill(black);
	text(str, x + bor * w, y + bor * h, x + w - 4*bor * w, y + h - 4*bor * h);
	
	fill(white);
	text(str, x + w + bor * w, y + bor * h, x + w - 4*bor * w, y + h - 4*bor * h);
}

// Determines which square (if any) was clicked
function mouseClicked() {
	var inBox1 = mouseX > x && mouseX < x + w && mouseY > y && mouseY < y + h;
	var inBox2 = mouseX > x + w && mouseX < x + 2*w && mouseY > y && mouseY < y + h;

	if (inBox1) {
		console.log("box1")
		col = genRandColor();
		// Send JSON to server
		postColor(1);
	} else if(inBox2) {
		console.log("box2")
		col = genRandColor();
		// Send JSON to server
		postColor(0);
	}

	return false;
}

function postColor(bw) {
	// ajax the JSON to the server
	resp = [red(col), green(col), blue(col), bw]; // black = 1, white = 0
	$.post("receiver", JSON.stringify(resp), function(){

	});

	// stop link reloading the page
	event.preventDefault();
}

// Rescales elements when window size changes
function windowResized() {
	resizeCanvas(windowWidth, windowHeight);
}